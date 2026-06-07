import streamDeck from "@elgato/streamdeck";
import { DaemonMonitor } from "./actions/daemon-monitor.js";
import { ForgeQueue } from "./actions/forge-queue.js";
import { ForgeStage } from "./actions/forge-stage.js";
import { GraphPending } from "./actions/graph-pending.js";
import { PipelineHealth } from "./actions/pipeline-health.js";
import { PipelineStart } from "./actions/pipeline-start.js";
import { TriggerWatch } from "./actions/trigger-watch.js";

streamDeck.actions.registerAction(new DaemonMonitor());
streamDeck.actions.registerAction(new ForgeQueue());
streamDeck.actions.registerAction(new ForgeStage());
streamDeck.actions.registerAction(new GraphPending());
streamDeck.actions.registerAction(new PipelineHealth());
streamDeck.actions.registerAction(new PipelineStart());
streamDeck.actions.registerAction(new TriggerWatch());
streamDeck.connect();
