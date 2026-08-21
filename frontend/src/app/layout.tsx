import type { Metadata } from "next";
import { Anybody, Inter, Noto_Serif } from "next/font/google";
import { Chrome } from "@/components/Chrome";
import "./globals.css";

// Fallbacks for the two commercial faces the design calls for.
// Anybody is a variable-width grotesque and stands in for Vanguard; Noto Serif
// stands in for Athelas, which resolves natively on macOS.
const anybody = Anybody({
  subsets: ["latin"],
  weight: ["400", "700", "800", "900"],
  style: ["normal", "italic"],
  variable: "--font-anybody",
});
const notoSerif = Noto_Serif({
  subsets: ["latin"],
  weight: ["400", "600"],
  style: ["normal", "italic"],
  variable: "--font-noto-serif",
});
const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Vanguard — Multimodal Content Moderation",
  description:
    "Decision support for human moderators. Detects harm in the relationship between image and text, and explains why.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${anybody.variable} ${notoSerif.variable} ${inter.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col bg-black">
        <Chrome>{children}</Chrome>
      </body>
    </html>
  );
}
