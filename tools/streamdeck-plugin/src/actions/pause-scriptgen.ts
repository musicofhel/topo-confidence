import {
	action,
	KeyDownEvent,
	SingletonAction,
	WillAppearEvent,
	WillDisappearEvent,
	DidReceiveSettingsEvent,
	type Action,
} from "@elgato/streamdeck";

import { renderMonitorSvg, svgToDataUri } from "../lib/renderer.js";

interface PauseSettings {
	serverUrl?: string;
}

interface ButtonState {
	action: Action<PauseSettings>;
	serverUrl: string;
	paused: boolean;
	readyFes: number | null;
	reachable: boolean;
	busy: boolean;
}

const POLL_MS = 3000;
const DEFAULT_URL = "http://localhost:9876";

/**
 * Pause / resume the token-spending autopilot phases (scriptgen + experiment)
 * while triage keeps digesting papers. Press = toggle; the key shows the live
 * state (red PAUSED / green RUNNING) and the READY-FE backlog so you can see the
 * queue grow while paused. Backed by the status server's /toggle-scriptgen +
 * /scriptgen-state endpoints.
 */
@action({ UUID: "com.topoconfidence.daemon-monitor.pause-scriptgen" })
export class PauseScriptgen extends SingletonAction<PauseSettings> {
	private buttons = new Map<string, ButtonState>();
	private pollTimer: ReturnType<typeof setInterval> | null = null;

	private mk(action: Action<PauseSettings>, s: PauseSettings): ButtonState {
		return {
			action,
			serverUrl: s.serverUrl || DEFAULT_URL,
			paused: false,
			readyFes: null,
			reachable: false,
			busy: false,
		};
	}

	override async onWillAppear(ev: WillAppearEvent<PauseSettings>): Promise<void> {
		const btn = this.mk(ev.action, ev.payload.settings);
		this.buttons.set(ev.action.id, btn);
		this.render(btn);
		void this.pollOne(btn);
		if (!this.pollTimer) this.startPolling();
	}

	override async onWillDisappear(ev: WillDisappearEvent<PauseSettings>): Promise<void> {
		this.buttons.delete(ev.action.id);
		if (this.buttons.size === 0) this.stopPolling();
	}

	override async onDidReceiveSettings(ev: DidReceiveSettingsEvent<PauseSettings>): Promise<void> {
		const btn = this.buttons.get(ev.action.id);
		if (btn) {
			btn.serverUrl = ev.payload.settings.serverUrl || DEFAULT_URL;
			btn.action = ev.action;
		} else {
			this.buttons.set(ev.action.id, this.mk(ev.action, ev.payload.settings));
		}
	}

	override async onKeyDown(ev: KeyDownEvent<PauseSettings>): Promise<void> {
		const btn = this.buttons.get(ev.action.id);
		if (!btn || btn.busy) return;
		btn.busy = true;
		try {
			const resp = await fetch(`${btn.serverUrl}/toggle-scriptgen`, {
				method: "POST",
				signal: AbortSignal.timeout(3000),
			});
			btn.reachable = resp.ok;
			// Optimistic flip so the key reacts instantly; the next poll reconciles.
			if (resp.ok) btn.paused = !btn.paused;
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
			const resp = await fetch(`${btn.serverUrl}/scriptgen-state`, {
				signal: AbortSignal.timeout(2000),
			});
			if (!resp.ok) throw new Error(String(resp.status));
			const data = (await resp.json()) as { paused?: boolean; ready_fes?: number | null };
			btn.paused = !!data.paused;
			btn.readyFes = typeof data.ready_fes === "number" ? data.ready_fes : null;
			btn.reachable = true;
		} catch {
			btn.reachable = false;
		}
		this.render(btn);
	}

	private render(btn: ButtonState): void {
		if (!btn.reachable) {
			const svg = renderMonitorSvg({ title: "SCR GEN", subtitle: "NO CONN", color: "red", flashOn: false });
			btn.action.setImage(svgToDataUri(svg));
			return;
		}
		const q = btn.readyFes == null ? "" : `q${btn.readyFes}`;
		const svg = renderMonitorSvg({
			title: btn.paused ? "PAUSED" : "RUNNING",
			subtitle: q,
			color: btn.paused ? "red" : "green",
			flashOn: true,
		});
		btn.action.setImage(svgToDataUri(svg));
	}
}
