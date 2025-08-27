
type ModelName = "sms" | "email";
type HealthStatus = "loading" | "ok" | "down";

interface HeaderProps {
    header: string;
    model: ModelName;
    onModelChange: (next: ModelName) => void;
    status: HealthStatus;
}

function Header({header,model,onModelChange, status}: HeaderProps) {
    function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
        onModelChange(e.target.value as ModelName);
  }
  return (
    <div>
        <h1>{header}</h1>
        <select id="model" value={model} onChange={handleChange}>
            <option value="sms">SMS</option>
            <option value="email">Email</option>
        </select>
        <p>{status}</p>
    </div>
  )
}

export default Header