import { describe, expect, it } from "bun:test";
import { convertMessages } from "@oh-my-pi/pi-ai/providers/openai-completions";
import {
	isSurvivingAssistant,
	shouldDropTruncatedThinkingOnlyAssistant,
	transformMessages,
} from "@oh-my-pi/pi-ai/providers/transform-messages";
import type { AssistantMessage, Message, Model } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";

/**
 * Regression test: interrupting a GLM-5.4 turn mid-thinking (Esc) used to drop
 * the partial CoT from every subsequent replay. `transformMessages` treated
 * thinking-only truncated turns (`stopReason: "aborted"`) as dead weight for
 * every target, but chat-completions reasoning hosts (GLM/Zhipu, DeepSeek,
 * …) replay thinking natively as `reasoning_content` — verified live against
 * GLM-5.4: the turn is accepted and the model recalls its interrupted chain.
 * The thinking-only turn must now survive for targets whose encoder writes
 * thinking back under `compat`, and must still drop for Anthropic (its
 * back-to-back assistant + unsigned-thinking contracts, pinned by
 * anthropic-thinking-only-length-truncated.test.ts) and when no target is
 * known.
 */

const glm: Model<"openai-completions"> = buildModel({
	api: "openai-completions",
	provider: "zhipu-coding-plan",
	id: "glm-5.4",
	name: "GLM-5.4",
	baseUrl: "https://open.bigmodel.cn/api/coding/paas/v4",
	input: ["text", "image"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	maxTokens: 131072,
	contextWindow: 1000000,
	reasoning: true,
});

const claude: Model<"anthropic-messages"> = buildModel({
	api: "anthropic-messages",
	provider: "anthropic",
	id: "claude-opus-4-7",
	name: "Claude Opus 4.7",
	baseUrl: "https://api.anthropic.com",
	input: ["text", "image"],
	cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
	maxTokens: 8192,
	contextWindow: 200000,
	reasoning: true,
});

function makeAbortedThinkingOnly(thinking: string, overrides: Partial<AssistantMessage> = {}): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "thinking", thinking }],
		api: glm.api,
		provider: glm.provider,
		model: glm.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "aborted",
		errorMessage: "Interrupted by user",
		timestamp: Date.now(),
		...overrides,
	};
}

function makeUser(text: string): Message {
	return { role: "user", content: text, timestamp: Date.now() };
}

describe("interrupted thinking-only turns replay for reasoning_content targets", () => {
	it("keeps the turn for GLM and ships it as reasoning_content with null content", () => {
		const messages: Message[] = [
			makeUser("investigate the stuck background command"),
			makeAbortedThinkingOnly("The user says the build is stuck. Let me check the nix log first..."),
			makeUser("there is nothing stuck"),
		];
		const transformed = transformMessages(messages, glm);
		expect(transformed.some(m => m.role === "assistant")).toBe(true);

		const wire = convertMessages(glm, { messages: transformed, tools: [] }, glm.compat);
		const assistant = wire.filter(m => m.role === "assistant");
		expect(assistant).toHaveLength(1);
		expect(assistant[0]?.content).toBe("");
		expect(assistant[0]?.reasoning_content).toContain("nix log first");
		expect(assistant[0]?.tool_calls).toBeUndefined();
	});

	it("still drops truly-empty truncated turns and turns with thinking for Anthropic", () => {
		const empty = makeAbortedThinkingOnly("", { content: [] });
		expect(shouldDropTruncatedThinkingOnlyAssistant(empty, glm)).toBe(true);

		const claudeTurn = makeAbortedThinkingOnly("partial chain", {
			api: claude.api,
			provider: claude.provider,
			model: claude.id,
		});
		expect(shouldDropTruncatedThinkingOnlyAssistant(claudeTurn, claude)).toBe(true);
		const transformed = transformMessages([makeUser("go"), claudeTurn, makeUser("again")], claude);
		expect(transformed.some(m => m.role === "assistant")).toBe(false);
	});

	it("keeps the strict predicate and non-survival when no target is known", () => {
		const turn = makeAbortedThinkingOnly("partial chain");
		expect(shouldDropTruncatedThinkingOnlyAssistant(turn)).toBe(true);
		expect(isSurvivingAssistant(turn)).toBe(false);
	});

	it("isSurvivingAssistant agrees with the transform when the target model is known", () => {
		const turn = makeAbortedThinkingOnly("partial chain");
		expect(isSurvivingAssistant(turn, glm)).toBe(true);
		const claudeTurn = makeAbortedThinkingOnly("partial chain", {
			api: claude.api,
			provider: claude.provider,
			model: claude.id,
		});
		expect(isSurvivingAssistant(claudeTurn, claude)).toBe(false);
	});

	it("keeps length-truncated thinking for GLM too", () => {
		const turn = makeAbortedThinkingOnly("ran out of tokens mid-reasoning", { stopReason: "length" });
		expect(shouldDropTruncatedThinkingOnlyAssistant(turn, glm)).toBe(false);
		expect(shouldDropTruncatedThinkingOnlyAssistant(turn, claude)).toBe(true);
	});
});
