import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const sampleForecast = [
  { year: 2025, baseline: 22000, withStrategy: 18000 },
  { year: 2026, baseline: 23000, withStrategy: 18500 },
  { year: 2027, baseline: 24000, withStrategy: 19000 },
];

const strategy = {
  name: "S-Corp Election",
  eligibility: "Net Schedule C income > $50K, active business involvement",
  savings: "$4,000–$6,000 annually in SE tax",
  risks: "Increased compliance costs, payroll requirements",
  steps: [
    "Form 2553 filed by March 15",
    "Setup payroll service",
    "Adjust Q1 estimated payments",
  ],
};

export default function TaxStrategyCard() {
  const [view, setView] = useState("summary");

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "auto" }}>
      <div
        style={{
          border: "1px solid #ddd",
          borderRadius: "12px",
          padding: "20px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
        }}
      >
        <h2 style={{ fontSize: "20px", fontWeight: "bold" }}>{strategy.name}</h2>
        <p style={{ color: "#555", fontSize: "14px" }}>
          Estimated Savings: {strategy.savings}
        </p>

        <div style={{ marginTop: "10px", fontSize: "14px" }}>
          <p>
            <strong>Eligibility:</strong> {strategy.eligibility}
          </p>
          <p>
            <strong>Risks:</strong> {strategy.risks}
          </p>
          <p>
            <strong>Steps to Implement:</strong>
          </p>
          <ul>
            {strategy.steps.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ul>
        </div>

        <div style={{ marginTop: "20px" }}>
          <button
            onClick={() => setView(view === "summary" ? "chart" : "summary")}
            style={{
              padding: "10px 16px",
              backgroundColor: "#2563eb",
              color: "white",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
            }}
          >
            {view === "summary" ? "View ROI Forecast" : "Back to Strategy"}
          </button>
        </div>
      </div>

      {view === "chart" && (
        <div
          style={{
            marginTop: "20px",
            border: "1px solid #ddd",
            borderRadius: "12px",
            padding: "20px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
          }}
        >
          <h3 style={{ fontSize: "16px", marginBottom: "10px" }}>
            3-Year ROI Forecast
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={sampleForecast}>
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="baseline"
                stroke="#8884d8"
                name="Without Strategy"
              />
              <Line
                type="monotone"
                dataKey="withStrategy"
                stroke="#82ca9d"
                name="With Strategy"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
