import "./globals.css";

export const metadata = {
  title: "Atlas — multi-agent researcher",
  description: "A multi-agent research orchestrator (LangGraph supervisor pattern).",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
