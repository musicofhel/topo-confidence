import streamDeck, {
	action,
	KeyDownEvent,
	SingletonAction,
	WillAppearEvent,
	WillDisappearEvent,
	DidReceiveSettingsEvent,
	type Action,
} from "@elgato/streamdeck";

import { poller, type StatusResponse, type PhaseStatus } from "../lib/poller.js";
import { renderSvg, svgToDataUri, type DaemonState } from "../lib/renderer.js";

interface DaemonSettings {
	phase?: string;
	serverUrl?: string;
	budgetCap?: number;
}

const FLASH_INTERVAL_MS = 800;

interface ButtonState {
	phase: string;
	currentState: DaemonState;
	currentPhaseData: PhaseStatus | null;
	animTimer: ReturnType<typeof setInterval> | null;
	flashOn: boolean;
	action: Action<DaemonSettings>;
}

@action({ UUID: "com.topoconfidence.daemon-monitor.phase" })
export class DaemonMonitor extends SingletonAction<DaemonSettings> {
	private buttons = new Map<string, ButtonState>();

	override async onWillAppear(ev: WillAppearEvent<DaemonSettings>): Promise<void> {
		const id = ev.action.id;
		const settings = ev.payload.settings;

		const btn: ButtonState = {
			phase: settings.phase ?? "triage",
			currentState: "no_conn",
			currentPhaseData: null,
			animTimer: null,
			flashOn: true,
			action: ev.action,
		};
		this.buttons.set(id, btn);

		poller.register(
			id,
			(data) => this.onStatusUpdate(id, data),
			settings.serverUrl || "http://localhost:9876",
			settings.budgetCap ?? 15,
		);

		this.renderStatic(btn);
	}

	override async onWillDisappear(ev: WillDisappearEvent<DaemonSettings>): Promise<void> {
		const id = ev.action.id;
		const btn = this.buttons.get(id);
		if (btn?.animTimer) clearInterval(btn.animTimer);
		poller.unregister(id);
		this.buttons.delete(id);
	}

	override async onDidReceiveSettings(ev: DidReceiveSettingsEvent<DaemonSettings>): Promise<void> {
		const id = ev.action.id;
		const settings = ev.payload.settings;
		let btn = this.buttons.get(id);

		if (!btn) {
			btn = {
				phase: settings.phase ?? "triage",
				currentState: "no_conn",
				currentPhaseData: null,
				animTimer: null,
				flashOn: true,
				action: ev.action,
			};
			this.buttons.set(id, btn);
		} else {
			btn.phase = settings.phase ?? "triage";
			btn.action = ev.action;
		}

		poller.unregister(id);
		poller.register(
			id,
			(data) => this.onStatusUpdate(id, data),
			settings.serverUrl || "http://localhost:9876",
			settings.budgetCap ?? 15,
		);
	}

	override async onKeyDown(ev: KeyDownEvent<DaemonSettings>): Promise<void> {
		const btn = this.buttons.get(ev.action.id);
		if (btn) {
			streamDeck.logger.info(`${btn.phase}: state=${btn.currentState}, item=${btn.currentPhaseData?.current_item ?? "none"}`);
		}
	}

	private onStatusUpdate(id: string, data: StatusResponse | null): void {
		const btn = this.buttons.get(id);
		if (!btn) return;

		if (!data) {
			btn.currentState = "no_conn";
			btn.currentPhaseData = null;
			this.stopAnimation(btn);
			this.renderStatic(btn);
			return;
		}

		const phaseData = data.daemons[btn.phase];
		if (!phaseData) {
			btn.currentState = "stopped";
			btn.currentPhaseData = null;
			this.stopAnimation(btn);
			this.renderStatic(btn);
			return;
		}

		btn.currentPhaseData = phaseData;
		const newState = phaseData.state as DaemonState;

		if (newState === "active" && btn.currentState !== "active") {
			btn.currentState = newState;
			this.startAnimation(btn);
		} else if (newState !== "active" && btn.currentState === "active") {
			btn.currentState = newState;
			this.stopAnimation(btn);
			this.renderStatic(btn);
		} else {
			btn.currentState = newState;
			if (newState !== "active") {
				this.renderStatic(btn);
			}
		}
	}

	private startAnimation(btn: ButtonState): void {
		this.stopAnimation(btn);
		btn.flashOn = true;
		this.renderStatic(btn);

		btn.animTimer = setInterval(() => {
			btn.flashOn = !btn.flashOn;
			const svg = renderSvg({
				state: btn.currentState,
				phase: btn.phase,
				flashOn: btn.flashOn,
			});
			btn.action.setImage(svgToDataUri(svg));
		}, FLASH_INTERVAL_MS);
	}

	private stopAnimation(btn: ButtonState): void {
		if (btn.animTimer) {
			clearInterval(btn.animTimer);
			btn.animTimer = null;
		}
	}

	private renderStatic(btn: ButtonState): void {
		const svg = renderSvg({
			state: btn.currentState,
			phase: btn.phase,
			flashOn: btn.flashOn,
		});
		btn.action.setImage(svgToDataUri(svg));
	}
}
