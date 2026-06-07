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

const STAGE_LABELS: Record<string, string> = {
	idle: "IDLE",
	scrape: "SCRAPE",
	claude: "CLAUDE",
	embed: "EMBED",
	store: "STORE",
	chunk: "CHUNK",
	bridge: "BRIDGE",
	done: "DONE",
	processing: "...",
};

interface ButtonState {
	animTimer: ReturnType<typeof setInterval> | null;
	flashOn: boolean;
	stage: string;
	action: Action<MonitorSettings>;
}

@action({ UUID: "com.topoconfidence.daemon-monitor.forge-stage" })
export class ForgeStage extends SingletonAction<MonitorSettings> {
	private buttons = new Map<string, ButtonState>();

	override async onWillAppear(ev: WillAppearEvent<MonitorSettings>): Promise<void> {
		const id = ev.action.id;
		const btn: ButtonState = { animTimer: null, flashOn: true, stage: "idle", action: ev.action };
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
			btn = { animTimer: null, flashOn: true, stage: "idle", action: ev.action };
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
			btn.stage = "err";
			this.render(btn);
			return;
		}

		const stage = data.forge_stage?.stage ?? "idle";
		const wasActive = btn.stage !== "idle" && btn.stage !== "err" && btn.stage !== "done";
		const isActive = stage !== "idle" && stage !== "done";
		btn.stage = stage;

		if (isActive && !wasActive) {
			this.startFlash(btn);
		} else if (!isActive && wasActive) {
			this.stopFlash(btn);
			this.render(btn);
		} else if (!isActive) {
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
		const subtitle = STAGE_LABELS[btn.stage] ?? btn.stage.toUpperCase();

		if (btn.stage === "err") {
			color = "red";
		} else if (btn.stage === "idle" || btn.stage === "done") {
			color = "green";
		} else {
			color = "yellow";
		}

		const svg = renderMonitorSvg({ title: "FORGE", subtitle, color, flashOn: btn.flashOn });
		btn.action.setImage(svgToDataUri(svg));
	}
}
