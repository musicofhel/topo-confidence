import type { DaemonState } from "./renderer.js";

export interface PhaseStatus {
	state: DaemonState;
	pid: number | null;
	budget_used: number;
	budget_cap: number;
	current_item: string | null;
	last_activity: string | null;
	completed_count?: number;
	completed_age?: number | null;
}

export interface StatusResponse {
	daemons: Record<string, PhaseStatus>;
}

type Listener = (data: StatusResponse | null) => void;

class Poller {
	private interval: ReturnType<typeof setInterval> | null = null;
	private listeners = new Map<string, Listener>();
	private serverUrl = "http://localhost:9876";
	private budgetCap = 15;
	private lastData: StatusResponse | null = null;

	register(id: string, callback: Listener, serverUrl?: string, budgetCap?: number): void {
		this.listeners.set(id, callback);
		if (serverUrl) this.serverUrl = serverUrl;
		if (budgetCap !== undefined) this.budgetCap = budgetCap;
		if (this.lastData) callback(this.lastData);
		if (!this.interval) this.start();
	}

	unregister(id: string): void {
		this.listeners.delete(id);
		if (this.listeners.size === 0) this.stop();
	}

	private start(): void {
		this.tick();
		this.interval = setInterval(() => this.tick(), 2000);
	}

	private stop(): void {
		if (this.interval) {
			clearInterval(this.interval);
			this.interval = null;
		}
	}

	private async tick(): Promise<void> {
		try {
			const resp = await fetch(`${this.serverUrl}/status?budget_cap=${this.budgetCap}`);
			if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
			this.lastData = (await resp.json()) as StatusResponse;
		} catch {
			this.lastData = null;
		}
		for (const cb of this.listeners.values()) {
			cb(this.lastData);
		}
	}
}

export const poller = new Poller();

export interface PipelineResponse {
	link_forge_queue: { pending: number; processing: number; completed: number; failed: number };
	forge_stage: { stage: string; title: string | null; queue_id: string | null };
	research_graph_pending: number;
	trigger_count: number;
	services: Record<string, boolean>;
}

type PipelineListener = (data: PipelineResponse | null) => void;

class PipelinePoller {
	private interval: ReturnType<typeof setInterval> | null = null;
	private listeners = new Map<string, PipelineListener>();
	private serverUrl = "http://localhost:9876";
	private lastData: PipelineResponse | null = null;

	register(id: string, callback: PipelineListener, serverUrl?: string): void {
		this.listeners.set(id, callback);
		if (serverUrl) this.serverUrl = serverUrl;
		if (this.lastData) callback(this.lastData);
		if (!this.interval) this.start();
	}

	unregister(id: string): void {
		this.listeners.delete(id);
		if (this.listeners.size === 0) this.stop();
	}

	private start(): void {
		this.tick();
		this.interval = setInterval(() => this.tick(), 3000);
	}

	private stop(): void {
		if (this.interval) {
			clearInterval(this.interval);
			this.interval = null;
		}
	}

	private async tick(): Promise<void> {
		try {
			const resp = await fetch(`${this.serverUrl}/pipeline`);
			if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
			this.lastData = (await resp.json()) as PipelineResponse;
		} catch {
			this.lastData = null;
		}
		for (const cb of this.listeners.values()) {
			cb(this.lastData);
		}
	}
}

export const pipelinePoller = new PipelinePoller();
