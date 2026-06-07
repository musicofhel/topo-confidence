import {
	action,
	SingletonAction,
	WillAppearEvent,
	WillDisappearEvent,
	DidReceiveSettingsEvent,
	type Action,
} from "@elgato/streamdeck";

import { pipelinePoller, type PipelineResponse } from "../lib/poller.js";
import { renderMonitorSvg, svgToDataUri } from "../lib/renderer.js";

interface MonitorSettings {
	serverUrl?: string;
}

const FLASH_INTERVAL_MS = 800;

interface ButtonState {
	animTimer: ReturnType<typeof setInterval> | null;
	flashOn: boolean;
	queueActive: number;
	action: Action<MonitorSettings>;
}

@action({ UUID: "com.topoconfidence.daemon-monitor.forge-queue" })
export class ForgeQueue extends SingletonAction<MonitorSettings> {
	private buttons = new Map<string, ButtonState>();

	override async onWillAppear(ev: WillAppearEvent<MonitorSettings>): Promise<void> {
		const id = ev.action.id;
		const btn: ButtonState = {
			animTimer: null,
			flashOn: true,
			queueActive: 0,
			action: ev.action,
		};
		this.buttons.set(id, btn);
		pipelinePoller.register(id, (data) => this.onUpdate(id, data), ev.payload.settings.serverUrl || "http://localhost:9876");
		this.render(btn);
	}

	override async onWillDisappear(ev: WillDisappearEvent<MonitorSettings>): Promise<void> {
		const id = ev.action.id;
		const btn = this.buttons.get(id);
		if (btn?.animTimer) clearInterval(btn.animTimer);
		pipelinePoller.unregister(id);
		this.buttons.delete(id);
	}

	override async onDidReceiveSettings(ev: DidReceiveSettingsEvent<MonitorSettings>): Promise<void> {
		const id = ev.action.id;
		let btn = this.buttons.get(id);
		if (!btn) {
			btn = { animTimer: null, flashOn: true, queueActive: 0, action: ev.action };
			this.buttons.set(id, btn);
		}
		pipelinePoller.unregister(id);
		pipelinePoller.register(id, (data) => this.onUpdate(id, data), ev.payload.settings.serverUrl || "http://localhost:9876");
	}

	private onUpdate(id: string, data: PipelineResponse | null): void {
		const btn = this.buttons.get(id);
		if (!btn) return;

		if (!data) {
			this.stopFlash(btn);
			btn.queueActive = -1;
			this.render(btn);
			return;
		}

		const active = data.link_forge_queue.pending + data.link_forge_queue.processing;
		const wasActive = btn.queueActive > 0;
		btn.queueActive = active;

		if (active > 0 && !wasActive) {
			this.startFlash(btn);
		} else if (active === 0 && wasActive) {
			this.stopFlash(btn);
			this.render(btn);
		} else if (active === 0) {
			this.render(btn);
		}
	}

	private startFlash(btn: ButtonState): void {
		this.stopFlash(btn);
		btn.flashOn = true;
		this.render(btn);
		btn.animTimer = setInterval(() => {
			btn.flashOn = !btn.flashOn;
			this.render(btn);
		}, FLASH_INTERVAL_MS);
	}

	private stopFlash(btn: ButtonState): void {
		if (btn.animTimer) {
			clearInterval(btn.animTimer);
			btn.animTimer = null;
		}
		btn.flashOn = true;
	}

	private render(btn: ButtonState): void {
		let color: "green" | "yellow" | "red";
		let subtitle: string;

		if (btn.queueActive < 0) {
			color = "red";
			subtitle = "ERR";
		} else if (btn.queueActive > 0) {
			color = "yellow";
			subtitle = String(btn.queueActive);
		} else {
			color = "green";
			subtitle = "CLEAR";
		}

		const svg = renderMonitorSvg({ title: "FORGE", subtitle, color, flashOn: btn.flashOn });
		btn.action.setImage(svgToDataUri(svg));
	}
}
