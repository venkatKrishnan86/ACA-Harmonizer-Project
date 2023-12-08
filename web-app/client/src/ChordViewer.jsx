import { useEffect, useState, useRef } from "react"
import styled from "styled-components"

const ChordViewerContainer = styled.div`
    display: flex;
    flex-direction: row;
    width: 100%;
    padding: 1rem;
    justify-content: center;
    align-items: flex-end;
    gap: 2rem;
`

const ChordContainer = styled.div`
    padding: 1rem;
    justify-content: center;
    align-items: center;
    border-radius: 0.5rem;
    text-align: center;
    width: 20%;
    background-color: ${props => props.isCurrent ? "lightGrey" : "grey"};
    font-size: ${props => props.isCurrent ? "6rem" : "3rem"};
    height: ${props => props.isCurrent ? "100%" : "50%"};
    box-sizing: border-box;
`

export default function ChordViewer({chords, isPlaying, editedTime}) {
    const [curTime, setCurrentTime] = useState(0)
    const [curChordIndex, setCurrentChordIndex] = useState(0)
    const [timer, setTimer] = useState(null)

    useEffect(() => {
        if (isPlaying) {
            let timer = startTimer()
            setTimer(timer)
            return () => clearInterval(timer)
        } else {
            stopTimer()
        }
    }, [isPlaying])

    useEffect(() => {
        console.log("Chords changed")
        stopTimer()
        setCurrentTime(0)
        setCurrentChordIndex(0)
    }, [chords])

    useEffect(() => {
        stopTimer()
        setCurrentTime(editedTime)
        var currentChordIndex = chords.findIndex((chord, index) => {
            if (chord.time > editedTime) {
                return true
            }
        })
        currentChordIndex = currentChordIndex == 0 ? 0 : currentChordIndex - 1
        setCurrentChordIndex(currentChordIndex)
        if (isPlaying) {
            startTimer()
        }
    }, [editedTime])

    const startTimer = () => {
        var curChordIndexRef = curChordIndex
        var curTimeRef = curTime
        const interval = setInterval(() => {
            if (chords && chords.length > 0 && curChordIndexRef < chords.length) {
                curTimeRef = curTimeRef + 0.1
                setCurrentTime(prevTime => prevTime + 0.1)
                if (chords[curChordIndexRef].time < curTimeRef) {
                    setCurrentChordIndex(curChordIndexRef)
                    curChordIndexRef = curChordIndexRef + 1
                }
            }
        }, 100)
        return interval
    }

    const stopTimer = () => {
        clearInterval(timer)
        setTimer(null)
    }

    return (
        <>
            <ChordViewerContainer>
                {chords && chords.map((chord, index) => {
                    let isShowChord = (index >= curChordIndex - 2) && index <= curChordIndex + 2
                    return isShowChord && <ChordContainer isCurrent={index == curChordIndex} key={index}>
                        {chord.chord}
                        {/* {chord.time} */}
                    </ChordContainer>
                })}
            </ChordViewerContainer>
        </>
    )
}