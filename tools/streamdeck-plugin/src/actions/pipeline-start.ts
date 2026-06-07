import {
	action,
	KeyDownEvent,
	SingletonAction,
	WillAppearEvent,
	WillDisappearEvent,
	DidReceiveSettingsEvent,
	type Action,
} from "@elgato/streamdeck";
import { exec } from "node:child_process";

import { renderMonitorSvg, svgToDataUri } from "../lib/renderer.js";

interface StartSettings {
	serverUrl?: string;
	startCommand?: string;
}

type ButtonMode = "up" | "down" | "starting";

interface ButtonState {
	mode: ButtonMode;
	action: Action<StartSettings>;
	serverUrl: string;
	startCommand: string;
}

const POLL_MS = 3000;

@action({ UUID: "com.topoconfidence.daemon-monitor.pipeline-start" })
export class PipelineStart extends SingletonAction<StartSettings> {
	private buttons = new Map<string, ButtonState>();
	private pollTimer: ReturnType<typeof setInterval> | null = null;

	override async onWillAppear(ev: WillAppearEvent<StartSettings>): Promise<void> {
		const id = ev.action.id;
		const s = ev.payload.settings;
		const btn: ButtonState = {
			mode: "down",
			action: ev.action,
			serverUrl: s.serverUrl || "http://localhost:9876",
			startCommand: s.startCommand || "wsl.exe bash ~/start-research-pipeline.sh",
		};
		this.buttons.set(id, btn);
		this.render(btn);
		if (!this.pollTimer) this.startPolling();
	}

	override async onWillDisappear(ev: WillDisappearEvent<StartSettings>): Promise<void> {
		this.buttons.delete(ev.action.id);
		if (this.buttons.size === 0) this.stopPolling();
	}

	override async onDidReceiveSettings(ev: DidReceiveSettingsEvent<StartSettings>): Promise<void> {
		const id = ev.action.id;
		const s = ev.payload.settings;
		let btn = this.buttons.get(id);
		if (!btn) {
			btn = {
				mode: "down",
				action: ev.action,
				serverUrl: s.serverUrl || "http://localhost:9876",
				startCommand: s.startCommand || "wsl.exe bash ~/start-research-pipeline.sh",
			};
			this.buttons.set(id, btn);
		} else {
			btn.serverUrl = s.serverUrl || "http://localhost:9876";
			btn.startCommand = s.startCommand || "wsl.exe bash ~/start-research-pipeline.sh";
			btn.action = ev.action;
		}
	}

	override async onKeyDown(ev: KeyDownEvent<StartSettings>): Promise<void> {
		const btn = this.buttons.get(ev.action.id);
		if (!btn) return;

		if (btn.mode === "up") return;
		if (btn.mode === "starting") return;

		btn.mode = "starting";
		this.render(btn);

		exec(btn.startCommand, { timeout: 120_000 }, (err) => {
			if (err) {
				btn.mode = "down";
				this.render(btn);
			}
		});
	}

	private startPolling(): void {
		this.poll();
		this.pollTimer = setInterval(() => this.poll(), POLL_MS);
	}

	private stopPolling(): void {
		if (this.pollTimer) {
			clearInterval(this.pollTimer);
			this.pollTimer = null;
		}
	}

	private async poll(): Promise<void> {
		const urls = new Set<string>();
		for (const btn of this.buttons.values()) urls.add(btn.serverUrl);

		const results = new Map<string, boolean>();
		for (const url of urls) {
			try {
				const resp = await fetch(`${url}/health`, { signal: AbortSignal.timeout(2000) });
				results.set(url, resp.ok);
			} catch {
				results.set(url, false);
			}
		}

		for (const btn of this.buttons.values()) {
			const reachable = results.get(btn.serverUrl) ?? false;
			const prev = btn.mode;
			if (reachable) {
				btn.mode = "up";
			} else if (prev !== "starting") {
				btn.mode = "down";
			}
			this.render(btn);
		}
	}

	private render(btn: ButtonState): void {
		let color: "green" | "yellow" | "red";
		let subtitle: string;

		switch (btn.mode) {
			case "up":
				color = "green";
				subtitle = "UP";
				break;
			case "starting":
				color = "yellow";
				subtitle = "WAIT";
				break;
			case "down":
				color = "red";
				subtitle = "START";
				break;
		}

		const svg = renderMonitorSvg({ title: "PIPE", subtitle, color, flashOn: true });
		btn.action.setImage(svgToDataUri(svg));
	}
}
