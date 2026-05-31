import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/Header";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "PriceIQ — Recomendador inteligente de precios",
  description: "Optimiza el precio de venta cuando el proveedor cambia el costo. "
    + "Basado en LightGBM, Item2Vec, Conformal Prediction y validación quasi-experimental.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-gradient-to-br from-slate-50 via-white to-indigo-50/30">
        <Header />
        {children}
      </body>
    </html>
  );
}
