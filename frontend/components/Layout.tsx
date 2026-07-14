import React from "react";
import Navbar from "./Navbar";

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div>
      <Navbar />
      <main style={{ maxWidth: "960px", margin: "2rem auto", padding: "0 1rem" }}>
        {children}
      </main>
    </div>
  );
}
