
type ModelName = "sms" | "email";
type HealthStatus = "loading" | "ok" | "down";

interface HeaderProps {
    header: string;
    model: ModelName;
    onModelChange: (next: ModelName) => void;
    status: HealthStatus;
}

function Header({ header, model, onModelChange, status }: HeaderProps) {
    function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
        onModelChange(e.target.value as ModelName);
    }

    const color =
        status === "ok" ? "green" : status === "down" ? "red" : "gray";
    const label =
        status === "ok" ? "Online" : status === "down" ? "Offline" : "Checking…";
    return (
        <header style={{ borderBottom: "1px solid #eee", padding: 12 }}>
            <div
                style={{
                    maxWidth: 900,
                    margin: "0 auto",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 12,
                }}
            >
                {/* Left: title */}
                <h1 style={{ fontSize: 18, fontWeight: 600 }}>{header}</h1>

                {/* Right: model picker + status */}
                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                    {/* Model selector */}
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <label htmlFor="model">Model:</label>
                        <select id="model" value={model} onChange={handleChange}>
                            <option value="sms">SMS</option>
                            <option value="email">Email</option>
                        </select>
                    </div>

                    {/* Status dot + text */}
                    <div
                        style={{ display: "flex", alignItems: "center", gap: 8 }}
                        aria-live="polite"
                    >
                        <span
                            // tiny colored circle
                            style={{
                                display: "inline-block",
                                width: 10,
                                height: 10,
                                borderRadius: "50%",
                                backgroundColor: color,
                            }}
                            aria-label={`API status: ${label}`}
                            title={`API status: ${label}`}
                        />
                        <span style={{ fontSize: 14, color: "#555" }}>{label}</span>
                    </div>
                </div>
            </div>
        </header>
    )
}

export default Header