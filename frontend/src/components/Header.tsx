
type ModelName = "sms" | "email";
type HealthStatus = "loading" | "ok" | "down";

interface HeaderProps {
    header: string;
    model: ModelName;
    onModelChange: (next: ModelName) => void;
    status: HealthStatus;
    className?: string;
}

function HealthDot({ status }: { status: HealthStatus }) {
    const colorClass =
        status === "ok" ? "bg-green-500" : status === "down" ? "bg-red-500" : "bg-gray-400";
    const label =
        status === "ok" ? "Online" : status === "down" ? "Offline" : "Checking…";

    return (
        <span className="inline-flex items-center gap-2" aria-live="polite">
            <span
                className={`h-2.5 w-2.5 rounded-full ${colorClass} ${status === "loading" ? "animate-pulse" : ""
                    }`}
                aria-label={`API status: ${label}`}
                title={`API status: ${label}`}
            />
            <span className="text-sm text-gray-600">{label}</span>
        </span>
    );
}


function Header({ header, model, onModelChange, status, className = "", }: HeaderProps) {
    function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
        onModelChange(e.target.value as ModelName);
    }

    return (
        <header className={`sticky top-0 z-40 border-b border-gray-200 bg-white/70 backdrop-blur ${className}`}>
            <div className="mx-auto flex max-w-5xl flex-col items-center gap-3 px-4 py-3 text-center md:flex-row md:items-center md:justify-between md:text-left">
                {/* Left: title */}
                <div className="flex items-center gap-2">
                    <span aria-hidden className="text-xl">📧</span>
                    <h1 className="text-lg font-semibold tracking-tight">{header}</h1>
                </div>

                {/* Right: model picker + status */}
                <div className="flex flex-col items-center gap-3 md:flex-row md:items-center md:gap-6">
                    {/* Model selector */}
                    <div className="flex items-center gap-2">
                        <label htmlFor="model" className="text-sm font-medium text-gray-700">
                            Model:
                        </label>
                        <select
                            id="model"
                            value={model}
                            onChange={handleChange}
                            className="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm outline-none focus:ring-2 focus:ring-blue-500"
                            aria-describedby="model-help"
                        >
                            <option value="sms">SMS</option>
                            <option value="email">Email</option>
                        </select>
                        <span id="model-help" className="sr-only">
                            Choose which model to use for predictions
                        </span>
                    </div>

                    {/* Health indicator */}
                    <HealthDot status={status} />
                </div>
            </div>
        </header>
    )
}

export default Header