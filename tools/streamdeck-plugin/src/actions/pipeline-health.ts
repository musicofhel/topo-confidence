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

interface ButtonState {
	downCount: number;
	action: Action<MonitorSettings>;
}

@action({ UUID: "com.topoconfidence.daemon-monitor.pipeline-health" })
export class PipelineHealth extends SingletonAction<MonitorSettings> {
	private buttons = new Map<string, ButtonState>();

	override async onWillAppear(ev: WillAppearEvent<MonitorSettings>): Promise<void> {
		const id = ev.action.id;
		const btn: ButtonState = { downCount: -1, action: ev.action };
		this.buttons.set(id, btn);
		pipelinePoller.register(id, (data) => this.onUpdate(id, data), ev.payload.settings.serverUrl || "http://localhost:9876");
		this.render(btn);
	}

	override async onWillDisappear(ev: WillDisappearEvent<MonitorSettings>): Promise<void> {
		const id = ev.action.id;
		pipelinePoller.unregister(id);
		this.buttons.delete(id);
	}

	override async onDidReceiveSettings(ev: DidReceiveSettingsEvent<MonitorSettings>): Promise<void> {
		const id = ev.action.id;
		let btn = this.buttons.get(id);
		if (!btn) {
			btn = { downCount: -1, action: ev.action };
			this.buttons.set(id, btn);
		}
		pipelinePoller.unregister(id);
		pipelinePoller.register(id, (data) => this.onUpdate(id, data), ev.payload.settings.serverUrl || "http://localhost:9876");
	}

	private onUpdate(id: string, data: PipelineResponse | null): void {
		const btn = this.buttons.get(id);
		if (!btn) return;

		if (!data) {
			btn.downCount = -1;
			this.render(btn);
			return;
		}

		btn.downCount = Object.values(data.services).filter((v) => !v).length;
		this.render(btn);
	}

	private render(btn: ButtonState): void {
		let color: "green" | "yellow" | "red";
		let subtitle: string;

		if (btn.downCount < 0) {
			color = "red";
			subtitle = "DOWN";
		} else if (btn.downCount === 0) {
			color = "green";
			subtitle = "OK";
		} else if (btn.downCount >= 5) {
			color = "red";
			subtitle = "DOWN";
		} else {
			color = "yellow";
			subtitle = `${btn.downCount} DOWN`;
		}

		const svg = renderMonitorSvg({ title: "PIPE", subtitle, color, flashOn: true });
		btn.action.setImage(svgToDataUri(svg));
	}
}
