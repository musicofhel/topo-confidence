import streamDeck, {
	action,
	KeyDownEvent,
	SingletonAction,
	WillAppearEvent,
	WillDisappearEvent,
	DidReceiveSettingsEvent,
	type Action,
} from "@elgato/streamdeck";

import { poller, type StatusResponse } from "../lib/poller.js";
import { renderMonitorSvg, svgToDataUri } from "../lib/renderer.js";

interface ProgressSettings {
	serverUrl?: string;
}

// A completion seen within this window means the experiment phase is actively
// chewing through FEs (the completed-marker file's mtime drives it server-side).
const FRESH_MS = 120_000;

interface ButtonState {
	action: Action<ProgressSettings>;
}

/**
 * Live experiment-phase FE progress for the Stream Deck sidebar.
 *
 *   line 1: the FE being run now (or the last one) e.g. "FE383"
 *   line 2: total FEs completed       e.g. "✓150"
 *   colour: green  = a run finished < 2 min ago (working through the queue)
 *           yellow = alive but no recent completion (queue drained / idle)
 *           red    = experiment daemon stopped or the status server is unreachable
 */
@action({ UUID: "com.topoconfidence.daemon-monitor.experiment-progress" })
export class ExperimentProgress extends SingletonAction<ProgressSettings> {
	private buttons = new Map<string, ButtonState>();

	override async onWillAppear(ev: WillAppearEvent<ProgressSettings>): Promise<void> {
		this.attach(ev.action.id, ev.action, ev.payload.settings);
	}

	override async onWillDisappear(ev: WillDisappearEvent<ProgressSettings>): Promise<void> {
		poller.unregister(ev.action.id);
		this.buttons.delete(ev.action.id);
	}

	override async onDidReceiveSettings(ev: DidReceiveSettingsEvent<ProgressSettings>): Promise<void> {
		poller.unregister(ev.action.id);
		this.attach(ev.action.id, ev.action, ev.payload.settings);
	}

	override async onKeyDown(ev: KeyDownEvent<ProgressSettings>): Promise<void> {
		streamDeck.logger.info("experiment-progress pressed");
	}

	private attach(id: string, action: Action<ProgressSettings>, settings: ProgressSettings): void {
		this.buttons.set(id, { action });
		poller.register(
			id,
			(data) => this.onStatusUpdate(id, data),
			settings.serverUrl || "http://localhost:9876",
		);
	}

	private onStatusUpdate(id: string, data: StatusResponse | null): void {
		const btn = this.buttons.get(id);
		if (!btn) return;

		const phase = data?.daemons?.experiment;
		if (!data || !phase || phase.state === "stopped") {
			this.render(btn, "EXP", "—", "red");
			return;
		}

		const count = phase.completed_count ?? 0;
		const age = phase.completed_age;
		const advancing = typeof age === "number" && age * 1000 < FRESH_MS;

		// Prefer the FE running right now; fall back to the FE in the last log line.
		const feSource = phase.current_item || phase.last_activity || "";
		const m = feSource.match(/FE\d+/);
		const fe = m ? m[0] : "idle";

		const color = advancing ? "green" : "yellow";
		this.render(btn, fe, `✓${count}`, color);
	}

	private render(btn: ButtonState, line1: string, line2: string, color: "green" | "yellow" | "red"): void {
		const svg = renderMonitorSvg({ title: line1, subtitle: line2, color, flashOn: true });
		btn.action.setImage(svgToDataUri(svg));
	}
}
