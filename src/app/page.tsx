import React from "react";
import { P5Container } from "@/components/p5container";
import { mySketch } from "@/components/p5container";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 className="text-4xl font-bold">Teste</h1>
      <P5Container sketch={mySketch} />
    </main>
  );
}
