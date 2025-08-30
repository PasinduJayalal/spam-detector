
type ScoreBarProps  = {
    value?: number | null;
    text?: string; 
};


function ScoreBar({value , text}: ScoreBarProps ) {
    
    const isValid = typeof value === "number" && Number.isFinite(value)
    const clamped01 = isValid ? Math.max(0, Math.min(1, value)) : 0
    const percentage = Math.round(clamped01 * 100)

    const label = text ?? "Spam score";
    const displayText = isValid ? `${percentage}%` : "—";
    
    return (
        <div className="flex flex-col gap-1">
            <div className="mt-1 text-sm text-gray-700">
                {label}: <strong>{displayText}</strong>
            </div>
            <div
                className="h-3 w-full rounded-full bg-gray-200"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={isValid ? percentage : undefined}
                aria-label={label}
            >
                <div
                    className="h-3 rounded-full bg-blue-600"
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    )
}

export default ScoreBar