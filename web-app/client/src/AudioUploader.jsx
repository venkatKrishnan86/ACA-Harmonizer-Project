import React from 'react';
import { useState } from 'react';
import styled from 'styled-components';

const AudioUploaderContainer = styled.label`
    padding: 1rem;
    font-size: 2rem;
    border: 2px solid lightGrey;
    color: lightGrey;
    border-radius: 0.5rem;
    cursor: pointer;
    text-align: center;
`

export default function AudioUploader({handlerSetAudioSrc, handlerSetChords, handlerSetIsLoading}) {
    const handleFileUpload = async (event) => {
        handlerSetIsLoading(true)
        const file = event.target.files[0];
        const formData = new FormData();
        formData.append("file", file);
        try {
            const response = await fetch("http://127.0.0.1:5000/run_model", {
                method: "POST",
                body: formData,
            });
            if (response.ok) {
                let data = await response.json();
                let chords = data[0]
                let timeStamps = data[1]
                handlerSetChords(chords.map((chord, index) => {
                    return {
                        chord: chord,
                        time: timeStamps[index]
                    }
                }))
                try {
                    let audioFileResponse = await fetch("http://127.0.0.1:5000/audio")
                    if (audioFileResponse.ok) {
                        let audioFile = await audioFileResponse.blob()
                        handlerSetAudioSrc(URL.createObjectURL(audioFile));
                        handlerSetIsLoading(false)
                    } else {
                        console.log("Audio file fetch failed.")
                        handlerSetIsLoading(false)
                    }
                } catch (error) {
                    console.error("Audio file Error:", error);
                    handlerSetIsLoading(false)
                }
            } else {
                console.log("File upload failed.");
                handlerSetIsLoading(false)
            }
        } catch (error) {
            console.error("Error:", error);
            handlerSetIsLoading(false)
        }
    }
    return (
        <AudioUploaderContainer for="inputTag">
            Upload your melody
            <input id="inputTag" type="file" style={{display: "none"}} onChange={handleFileUpload}/>
        </AudioUploaderContainer>
    )
}