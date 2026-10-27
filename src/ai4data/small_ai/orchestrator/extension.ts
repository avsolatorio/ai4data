/**
 * pi extension: the tools of the ai4data.small_ai feedback loop.
 *
 * The agent that loads this extension gets seven tools and no others (run.sh starts pi with
 * --no-builtin-tools and an allowlist). Each tool runs one command of the harness,
 * `python -m ai4data.small_ai <command> <run folder>`, for the run folder fixed by the operator in
 * HARNESS_RUN. The agent cannot run a shell, read or write files, or choose another run folder.
 *
 *   harness_status   rounds so far, feedback scores, spending, and the budget
 *   gap_report       the gap report of a round: weak and over-assigned labels, confusions, thin labels
 *   find_labels      labels whose code or name matches a query, with their definitions
 *   label_examples   generated examples of a label from the accepted rounds
 *   label_errors     feedback texts the model got wrong for a label (only if the task allows it)
 *   run_round        run the next round with a targeted plan; returns the decision and the new scores
 *   stop_run         end the rounds, with the reason
 *
 * Environment: HARNESS_PYTHON (the Python interpreter with the harness installed or on PYTHONPATH)
 * and HARNESS_RUN (the run folder).
 */

import { execFile } from "node:child_process";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PYTHON = process.env.HARNESS_PYTHON ?? "python";
const RUN = process.env.HARNESS_RUN ?? "";

function harness(args: string[], timeoutMs = 60_000): Promise<string> {
	if (!RUN) return Promise.reject(new Error("HARNESS_RUN is not set"));
	return new Promise((resolve, reject) => {
		execFile(
			PYTHON,
			["-m", "ai4data.small_ai", ...args],
			{ env: process.env, maxBuffer: 32 * 1024 * 1024, timeout: timeoutMs },
			(error, stdout, stderr) => {
				if (error) {
					const tail = (stderr || String(error)).split("\n").slice(-12).join("\n");
					reject(new Error(tail));
				} else {
					resolve(stdout);
				}
			},
		);
	});
}

function text(body: string, limit = 24_000) {
	const shown = body.length > limit ? `${body.slice(0, limit)}\n[truncated at ${limit} characters]` : body;
	return { content: [{ type: "text" as const, text: shown }], details: undefined };
}

const status = defineTool({
	name: "harness_status",
	label: "Harness status",
	description: "Rounds done so far, the feedback score of each, whether it was accepted, the spending, and the budget.",
	parameters: Type.Object({}),
	async execute() {
		return text(await harness(["status", RUN]));
	},
});

const gap = defineTool({
	name: "gap_report",
	label: "Gap report",
	description:
		"The gap report of a round (default: the latest): the weakest labels with support, precision, recall, F1, " +
		"training examples, and labeller agreement; over-assigned labels; the most frequent confusions; and the labels " +
		"with the fewest training examples. Scores are on the feedback set of real texts.",
	parameters: Type.Object({ round: Type.Optional(Type.Integer({ minimum: 0 })) }),
	async execute(_id, params) {
		const args = ["gap", RUN];
		if (params.round !== undefined) args.push("--round", String(params.round));
		return text(await harness(args));
	},
});

const findLabels = defineTool({
	name: "find_labels",
	label: "Find labels",
	description: "Labels whose code starts with the query or whose name contains it, with their definitions (at most 25).",
	parameters: Type.Object({ query: Type.String({ minLength: 1 }) }),
	async execute(_id, params) {
		return text(await harness(["labels", RUN, "--query", params.query]));
	},
});

const examples = defineTool({
	name: "label_examples",
	label: "Label examples",
	description: "A sample of the generated training examples of one label, from the accepted rounds.",
	parameters: Type.Object({
		label: Type.String({ description: "The label id, as in the gap report" }),
		n: Type.Optional(Type.Integer({ minimum: 1, maximum: 15 })),
	}),
	async execute(_id, params) {
		return text(await harness(["examples", RUN, "--label", params.label, "--n", String(params.n ?? 8)]));
	},
});

const errors = defineTool({
	name: "label_errors",
	label: "Label errors",
	description:
		"Real feedback texts on which the latest accepted model missed a label or assigned it wrongly. Refused when the " +
		"task does not allow real texts to be shown to the orchestrator.",
	parameters: Type.Object({
		label: Type.String(),
		n: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
	}),
	async execute(_id, params) {
		return text(await harness(["errors", RUN, "--label", params.label, "--n", String(params.n ?? 6)]));
	},
});

const item = Type.Object({
	labels: Type.Array(Type.String(), { minItems: 1, description: "Label ids. A contrast item needs two or more." }),
	requests: Type.Integer({ minimum: 1, description: "Number of requests for this item; each request yields about 20 examples." }),
	mode: Type.Union([Type.Literal("positive"), Type.Literal("contrast"), Type.Literal("background")]),
	note: Type.String({
		minLength: 10,
		description: "Why this item is in the plan, and what the examples should cover. The writer sees this note.",
	}),
});

const round = defineTool({
	name: "run_round",
	label: "Run round",
	description:
		"Run the next round with this plan: generate the examples, check them, train a new model on all accepted " +
		"examples, and score it on the feedback set. The requests of all items must add up exactly to the round's " +
		"number of requests (see harness_status). Takes several minutes. Returns the decision (accepted or rejected) " +
		"and the new scores.",
	parameters: Type.Object({
		items: Type.Array(item, { minItems: 1 }),
		note: Type.String({ minLength: 10, description: "The reasoning behind the plan as a whole." }),
	}),
	async execute(_id, params) {
		const path = join(RUN, "next_plan.json");
		writeFileSync(path, JSON.stringify({ items: params.items, note: params.note }, null, 2));
		return text(await harness(["round", RUN, "--plan", path], 3 * 60 * 60 * 1000));
	},
});

const stop = defineTool({
	name: "stop_run",
	label: "Stop run",
	description: "End the rounds of this run. Give the reason, with the scores that support it.",
	parameters: Type.Object({ reason: Type.String({ minLength: 10 }) }),
	async execute(_id, params) {
		return { ...text(await harness(["stop", RUN, "--reason", params.reason])), terminate: true };
	},
});

export default function (pi: ExtensionAPI) {
	for (const tool of [status, gap, findLabels, examples, errors, round, stop]) pi.registerTool(tool);
}
