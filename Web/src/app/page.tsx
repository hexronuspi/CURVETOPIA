import DrawingCanvas from './bezier/canvas';
import Link from 'next/link';

export default function Home() {
  return (
    <main className="p-8 font-sans">
      <header className="text-center mb-10">
        <h1 className="text-5xl font-extrabold text-gray-800">CURVETOPIA</h1>
        <div className="mt-5 text-lg text-gray-700">
          <p className="font-semibold">Team:</p>
          <p>Sakshi Kumari</p>
          <p>Aditya Raj (Team Lead)</p>
          <p className="mt-2">
            <span className="font-semibold">Email (Team Lead): </span>
            <a
              href="mailto:adityar.ug22.ec@nitp.ac.in"
              className="text-blue-500 hover:underline"
            >
              adityar.ug22.ec@nitp.ac.in
            </a>
          </p>
          <p className="mt-2">
          GitHub: <span>
          <Link
              href="https://github.com/hexronuspi/CURVETOPIA"
              className="text-blue-500 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
             https://github.com/hexronuspi/CURVETOPIA
            </Link>
          </span>
            
          </p>
        </div>
      </header>
      <section className="flex justify-center">
        <DrawingCanvas />
      </section>
    </main>
  );
}
