import {
	action,
	KeyDownEvent,
	SingletonAction,
	WillAppearEvent,
	WillDisappearEvent,
	DidReceiveSettingsEvent,
	type Action,
} from "@elgato/streamdeck";

import { svgToDataUri } from "../lib/renderer.js";

interface ConfgateSettings {
	serverUrl?: string;
}

interface ButtonState {
	action: Action<ConfgateSettings>;
	serverUrl: string;
	pending: number | null;
	digesting: boolean;
	reachable: boolean;
	busy: boolean;
	flashOn: boolean;
	flashTimer: ReturnType<typeof setInterval> | null;
}

const POLL_MS = 3000;
const FLASH_INTERVAL_MS = 600;
const DEFAULT_URL = "http://localhost:9876";

const SIZE = 144;
const CX = SIZE / 2;

// confgate has its own palette — orange means "papers waiting to digest", which
// the shared green/yellow/red renderer doesn't carry, so this action draws its
// own SVG.
const COL = {
	orange: "#ea580c",       // pending > 0, idle — press me
	yellowOn: "#eab308",     // digesting (flash high)
	yellowOff: "#854d0e",    // digesting (flash low)
	green: "#16a34a",        // queue empty
	red: "#dc2626",          // server unreachable
};

function svg(bg: string, count: string, textColor: string): string {
	return `<svg xmlns="http://www.w3.org/2000/svg" width="${SIZE}" height="${SIZE}">
<rect width="${SIZE}" height="${SIZE}" rx="12" fill="${bg}"/>
<text x="${CX}" y="34" text-anchor="middle" font-family="Arial,sans-serif" font-size="22" font-weight="bold" fill="${textColor}">CONFGATE</text>
<text x="${CX}" y="108" text-anchor="middle" font-family="Arial,sans-serif" font-size="64" font-weight="bold" fill="${textColor}">${count}</text>
</svg>`;
}

/**
 * confgate paper-triage button. Shows the count of papers admitted to the
 * confgate research-graph that are still waiting to be deep-triaged (pending,
 * no brief yet):
 *   - ORANGE + count  → papers queued, press to digest
 *   - YELLOW blinking → digesting (triage_pending.sh running)
 *   - GREEN "0"       → queue empty, nothing to do
 *   - RED "?"         → status server unreachable
 * Press launches the digest; the queue drains to 0 as briefs land.
 * Backed by the status server's /confgate-state + /confgate-digest endpoints.
 */
@action({ UUID: "com.topoconfidence.daemon-monitor.confgate" })
export class ConfgateTriage extends SingletonAction<ConfgateSettings> {
	private buttons = new Map<string, ButtonState>();
	private pollTimer: ReturnType<typeof setInterval> | null = null;

	private mk(action: Action<ConfgateSettings>, s: ConfgateSettings): ButtonState {
		return {
			action,
			serverUrl: s.serverUrl || DEFAULT_URL,
			pending: null,
			digesting: false,
			reachable: false,
			busy: false,
			flashOn: true,
			flashTimer: null,
		};
	}

	override async onWillAppear(ev: WillAppearEvent<ConfgateSettings>): Promise<void> {
		const btn = this.mk(ev.action, ev.payload.settings);
		this.buttons.set(ev.action.id, btn);
		this.render(btn);
		void this.pollOne(btn);
		if (!this.pollTimer) this.startPolling();
	}

	override async onWillDisappear(ev: WillDisappearEvent<ConfgateSettings>): Promise<void> {
		const btn = this.buttons.get(ev.action.id);
		if (btn) this.stopFlash(btn);
		this.buttons.delete(ev.action.id);
		if (this.buttons.size === 0) this.stopPolling();
	}

	override async onDidReceiveSettings(ev: DidReceiveSettingsEvent<ConfgateSettings>): Promise<void> {
		const btn = this.buttons.get(ev.action.id);
		if (btn) {
			btn.serverUrl = ev.payload.settings.serverUrl || DEFAULT_URL;
			btn.action = ev.action;
		} else {
			this.buttons.set(ev.action.id, this.mk(ev.action, ev.payload.settings));
		}
	}

	override async onKeyDown(ev: KeyDownEvent<ConfgateSettings>): Promise<void> {
		const btn = this.buttons.get(ev.action.id);
		if (!btn || btn.busy) return;
		btn.busy = true;
		try {
			const resp = await fetch(`${btn.serverUrl}/confgate-digest`, {
				method: "POST",
				signal: AbortSignal.timeout(3000),
			});
			btn.reachable = resp.ok;
			if (resp.ok) {
				// Optimistic: start blinking instantly; the next poll reconciles.
				btn.digesting = true;
			}
		} catch {
			btn.reachable = false;
		} finally {
			btn.busy = false;
			this.render(btn);
		}
		void this.pollOne(btn);
	}

	private startPolling(): void {
		this.pollTimer = setInterval(() => {
			for (const btn of this.buttons.values()) void this.pollOne(btn);
		}, POLL_MS);
	}

	private stopPolling(): void {
		if (this.pollTimer) {
			clearInterval(this.pollTimer);
			this.pollTimer = null;
		}
	}

	private async pollOne(btn: ButtonState): Promise<void> {
		try {
			const resp = await fetch(`${btn.serverUrl}/confgate-state`, {
				signal: AbortSignal.timeout(2000),
			});
			if (!resp.ok) throw new Error(String(resp.status));
			const data = (await resp.json()) as {
				pending?: number | null;
				digesting?: boolean;
			};
			btn.pending = typeof data.pending === "number" ? data.pending : null;
			btn.digesting = !!data.digesting;
			btn.reachable = true;
		} catch {
			btn.reachable = false;
		}
		this.render(btn);
	}

	private startFlash(btn: ButtonState): void {
		if (btn.flashTimer) return;
		btn.flashOn = true;
		btn.flashTimer = setInterval(() => {
			btn.flashOn = !btn.flashOn;
			this.paint(btn);
		}, FLASH_INTERVAL_MS);
	}

	private stopFlash(btn: ButtonState): void {
		if (btn.flashTimer) {
			clearInterval(btn.flashTimer);
			btn.flashTimer = null;
		}
	}

	/** Decide flashing vs static, then paint once. */
	private render(btn: ButtonState): void {
		if (btn.reachable && btn.digesting) {
			this.startFlash(btn);
		} else {
			this.stopFlash(btn);
		}
		this.paint(btn);
	}

	/** Draw the current frame from state (+ flashOn during digestion). */
	private paint(btn: ButtonState): void {
		let bg: string;
		let count: string;
		let textColor = "#ffffff";
		if (!btn.reachable) {
			bg = COL.red;
			count = "?";
		} else if (btn.digesting) {
			bg = btn.flashOn ? COL.yellowOn : COL.yellowOff;
			count = btn.pending == null ? "···" : String(btn.pending);
			textColor = btn.flashOn ? "#000000" : "#ffffff";
		} else if (btn.pending && btn.pending > 0) {
			bg = COL.orange;
			count = String(btn.pending);
		} else {
			bg = COL.green;
			count = btn.pending == null ? "·" : String(btn.pending);
		}
		btn.action.setImage(svgToDataUri(svg(bg, count, textColor)));
	}
}
