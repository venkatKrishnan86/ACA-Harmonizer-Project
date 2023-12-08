import logo from './logo.svg';
import './App.css';
import AudioUploader from './AudioUploader';
import AudioPlayer from './AudioPlayer';
import ChordViewer from './ChordViewer';
import styled from 'styled-components';
import { useState } from 'react';
import ScaleLoader from "react-spinners/ScaleLoader";

const Parent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 1rem;
  height: 100vh;
  gap: 1rem;
  background-color: #282c34;
`
const Heading = styled.div`
  font-size: 20rem;
  color: lightGrey;
`

const AudioPlayerStyle = styled.div`
  // position: fixed;
  // bottom: 0;
`
function App() {
  const [audioSrc, setAudioSrc] = useState(null);
  const [chords, setChords] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [editedTime, setTimerChange] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  const handleTimerChange = (event) => {
    setTimerChange(event.target.currentTime);
  }

  return (
    <Parent>
      <Heading>Chordinator</Heading>
      {isLoading && 
        <ScaleLoader
          color={"#fff"}
          loading={isLoading}
          height={100}
          width={32}
          radius={2}
          margin={2}
        />
      }
      {
        !isLoading &&
        <>
          <AudioUploader handlerSetAudioSrc={setAudioSrc} handlerSetChords={setChords} handlerSetIsLoading={setIsLoading}/>
          {audioSrc && 
            <>
              <hr style={{width: "100%"}}/>
              <ChordViewer chords={chords} isPlaying={isPlaying} editedTime={editedTime}/>
              <AudioPlayerStyle><AudioPlayer audioSrc={audioSrc} handleIsPlaying={setIsPlaying} handleTimerChange={handleTimerChange}/></AudioPlayerStyle>
              </>
            }
        </>
      } 
    </Parent>
  );
}

export default App;
