import ScoreBar from "./ScoreBar";

type ModelName = "sms" | "email";

interface ResultPanelProps {
    model: ModelName;
    //loading: boolean;
    result: { label?: string; score?: number } | null;
    //error: string | null;
}




function ResultPanel({model , result}: ResultPanelProps) {

    //const resultNull = result === null ? "Paste a message and click Predict to see results." : "";
    const normalized = result?.label?.trim().toLowerCase() ?? "";;
    const isSpam = normalized === "spam";
    const isNotSpam = normalized === "ham" || normalized === "not_spam" || normalized === "not spam";
    const displayModel =  model.toUpperCase() ;
    



  return (
    <div>
        <div className="mx-auto max-w-5xl p-4">
            <h2 className="mb-2 text-lg font-medium text-gray-900">Result</h2>
            <div className="rounded-lg border border-gray-300 bg-gray-50 p-4">
                {!result ? (
                    <p className="text-gray-600">Paste a message and click Predict to see results</p>
                ) : (
                    <>
                        <p className="mb-2 text-gray-700">
                            Model: <strong className="uppercase">{displayModel}</strong>
                        </p>
                        {/* {isSpam ? (
                            <p className="mb-2 text-gray-700">
                                Prediction:{" "}
                                <strong className={normalized === "spam" ? "text-red-600" : "text-green-600"}>
                                    {result?.label ?? "—"}
                                </strong>
                            </p>
                        ) : (
                            <p className="mb-2 text-gray-700">
                                Prediction: <strong>{result?.label ?? "—"}</strong>
                            </p>
                        )} */}
                        {isSpam ? (
                            <p className="mb-2 text-gray-700">
                                Prediction:{" "}
                                <strong className="text-red-600">
                                    {result?.label ?? "—"}
                                </strong>
                            </p>
                        ) : isNotSpam ? (
                            <p className="mb-2 text-gray-700">
                                Prediction:{" "}
                                <strong className="text-green-600">
                                    {result?.label ?? "—"}
                                </strong>
                            </p>
                        ) : (
                            <p className="mb-2 text-gray-700">
                                Prediction: <strong>{result?.label ?? "—"}</strong>
                            </p>
                        )}
                        <div className="mt-4">
                            <p className="mb-1 text-gray-700">Spam Score:</p>
                            
                            <ScoreBar value={result?.score ?? null} text={undefined} />
                            
                        </div>
                    </>
                )}
            </div>
        </div>
    </div>
  )
}

export default ResultPanel