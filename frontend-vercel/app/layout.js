import "./globals.css";

export const metadata = {
  title: "Land Title Diligence — Generate Opinion",
  description: "Upload title documents and generate a legal opinion PDF",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
