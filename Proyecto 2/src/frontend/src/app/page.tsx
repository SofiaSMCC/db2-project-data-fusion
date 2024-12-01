'use client';
import { useEffect, useState } from 'react';
import Editor from 'react-simple-code-editor';
import Prism from 'prismjs';
import 'prismjs/components/prism-sql';
import 'prismjs/themes/prism-tomorrow.css';
import Table from './components/Table';
import Notification from './components/Notification';
import StopWords from './utils/stopwords';

export default function Home() {
  const [data, setData] = useState([]);
  const [query, setQuery] = useState(`SELECT * WHERE lyrics @@ "In a haze, a stormy haze" USING Spimi LIMIT 10 \nSELECT * WHERE lyrics @@ "In a haze, a stormy haze" USING PostgreSQL LIMIT 10`);
  const [time, setTime] = useState<number | null>(null);
  const [notification, setNotification] = useState<string | null>(null);
  const [lyrics, setLyrics] = useState<any | null>(null);
  const [showLyrics, setShowLyrics] = useState<boolean>(false);

  const executeQuery = (queryToExecute: string | undefined) => {
    if (!queryToExecute) {
      console.log("Query is empty.")
      return;
    }
    setData([]);
    const startTime = performance.now();

    console.log("Executing Query: ", queryToExecute);
    fetch('http://0.0.0.0:8000/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        "query": queryToExecute
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data);

        if (data["message"]) {
          setNotification(data["message"]);
        } else {
          setData(data);
        }
        const endTime = performance.now();
        setTime(endTime - startTime);
      })
      .catch((error) => {
        console.error('Error:', error);
        setNotification(error.message);
      });
  };

  const showLyricsModal = (lyricsText: any) => {
    setLyrics(lyricsText);
    setShowLyrics(true);
  };

  const hideLyricsModal = () => {
    setShowLyrics(false);
    setLyrics(null);
  };

  return (
    <main className={`flex flex-col max-w-7xl gap-3 p-20 mx-auto`}>
      {showLyrics && (
        <div
          className="fixed inset-0 z-20 flex items-center justify-center bg-black bg-opacity-50"
          onClick={hideLyricsModal}
        >
          <div
            className="relative bg-white w-2/5 h-3/4 rounded-md p-3 overflow-auto shadow-lg"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-4 flex flex-col items-center gap-4 text-gray-800 whitespace-pre-wrap">
              <h1 className='font-bold'>
                {lyrics?.song} - {lyrics?.artist}
              </h1>
              <div className='flex flex-col text-sm items-center'>
                {lyrics?.text.split('\n').map((line: string, index: number) => (
                  <p key={index}>
                    {line.split(' ').map((word, wordIndex) => {
                      const highlightWords = query?.match(/"(.*?)"/)?.[1]
                          .replace(/[^\w\s]/g, '')
                          .toLowerCase()
                          .split(" ")
                          .filter((queryWord) => !StopWords.includes(queryWord)) || [];
                      console.log(highlightWords);
                      const isHighlighted = highlightWords.includes(word.toLowerCase());

                      return isHighlighted ? (
                        <span key={wordIndex}>
                          {isHighlighted ? (
                            <span className="bg-yellow-300">{word}</span>
                          ) : (
                            word
                          )}
                          {" "}
                        </span>
                      ) : (
                        <span key={wordIndex}>{word} </span>
                      );
                    })}
                  </p>
                ))}

              </div>
            </div>
          </div>
        </div>
      )}

      <section className="flex flex-row justify-between w-full rounded-md p-3 border">
        <div className="flex gap-3">
          <p>Lyrics Finder</p>
        </div>
        <div className="flex">
          <p>v3.1.5</p>
        </div>
      </section>

      <section className="flex flex-col rounded-md p-3 gap-3 border">
        <section className="flex min-h-32 h-fit rounded-md p-2 border">
          <Editor
            value={query}
            onValueChange={code => setQuery(code)}
            highlight={code => Prism.highlight(code, Prism.languages.sql, 'sql')}
            padding={4}
            placeholder='Escribe tu consulta aqui...'
            style={{
              fontFamily: '"Fira code", "Fira Mono", monospace',
              fontSize: 12,
              width: '100%',
            }}
          />
        </section>

        <section className="flex flex-row gap-3 justify-between text-sm rounded-md p-3 border">
          <div className='flex align-middle'>
            {notification && (
              <Notification
                message={notification}
                onClose={() => setNotification(null)}
              />
            )}
          </div>
          <div className='flex flex-row gap-3'>
            <button className="border p-2 rounded-sm bg-gray-50 hover:bg-gray-100"
              onClick={() => executeQuery(window.getSelection()?.toString())}>Execute Selected</button>
            <button className="border p-2 rounded-sm bg-gray-50 hover:bg-gray-100"
              onClick={() => executeQuery(query)}>Execute Query</button>
          </div>
        </section>

        <section className="flex flex-col border rounded-md p-3">
          <Table data={data} time={time} onLyricsClick={(lyricsText: any) => showLyricsModal(lyricsText)} />
        </section>
      </section>

    </main>
  );
}
