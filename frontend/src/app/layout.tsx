import type { Metadata } from "next";
import { Anybody, Inter, Noto_Serif } from "next/font/google";
import { Nav } from "@/components/Nav";
import "./globals.css";

// The tri-font hierarchy from the Sentinel Noir design system: Anybody for
// technical labels and display numerics, Noto Serif for editorial headings so
// findings read as a formal record, Inter for body copy.
const anybody = Anybody({
  subsets: ["latin"],
  weight: ["400", "700", "800", "900"],
  variable: "--font-anybody",
});
const notoSerif = Noto_Serif({
  subsets: ["latin"],
  weight: ["400", "600"],
  variable: "--font-noto-serif",
});
const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Sentinel — Multimodal Content Moderation",
  description:
    "Decision support for human moderators. Detects harm in the relationship between image and text, and explains why.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${anybody.variable} ${notoSerif.variable} ${inter.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col">
        <div
          className="ambient-field pointer-events-none fixed inset-0 -z-10"
          aria-hidden
        />
        <Nav />
        <main className="mx-auto w-full max-w-[1440px] flex-1 px-6 py-10 md:px-12">
          {children}
        </main>
      </body>
    </html>
  );
}
