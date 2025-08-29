import React from "react";

type ModelName = "sms" | "email";

type PredictorFormProps = {
    model: ModelName;
    maxTextLen: number;
    loading: boolean;
    onSubmit: (text: string) => void;
};

export default function PredictorForm({
    model,
    maxTextLen,
    loading,
    onSubmit,
}: PredictorFormProps) {
    const [text, setText] = React.useState("");

    const count = text.length;
    const over = count > maxTextLen;
    const canSubmit = !loading && text.trim().length > 0 && !over;

    function handleSubmit(e: React.FormEvent) {
        e.preventDefault();
        if (!canSubmit) return;
        onSubmit(text.trim());
    }

    function handleClear() {
        setText("");
    }

    return (
        <form onSubmit={handleSubmit} className="mx-auto max-w-5xl p-4">
            {/* little model note */}
            <p className="mb-2 text-sm text-gray-600">
                Model: <strong className="uppercase">{model}</strong>
            </p>

            <label htmlFor="message" className="mb-1 block text-sm font-medium text-gray-700">
                Message
            </label>

            <textarea
                id="message"
                name="message"
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={6}
                placeholder="Paste a single SMS or email body…"
                aria-describedby="message-counter"
                aria-invalid={over || undefined}
                className={`w-full rounded-lg border px-3 py-2 shadow-sm outline-none focus:ring-2 focus:ring-blue-500 ${over ? "border-red-400" : "border-gray-300"
                    }`}
                disabled={loading}
            />

            <div id="message-counter" className="mt-1 flex items-center justify-between text-sm">
                <span className={over ? "text-red-600" : "text-gray-500"}>
                    {count} / {maxTextLen}
                </span>
                {over && <span className="text-red-600">Too long — shorten your text</span>}
            </div>

            <div className="mt-4 flex gap-2">
                <button
                    type="submit"
                    disabled={!canSubmit}
                    className="inline-flex items-center rounded-lg bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
                >
                    {loading ? "Predicting…" : "Predict"}
                </button>
                <button
                    type="button"
                    onClick={handleClear}
                    className="rounded-lg border px-4 py-2 hover:bg-gray-50"
                    disabled={loading}
                >
                    Clear
                </button>
            </div>
        </form>
    );
}
