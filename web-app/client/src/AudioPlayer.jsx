import ReactAudioPlayer from 'react-audio-player';

export default function AudioPlayer({audioSrc, handleIsPlaying, handleTimerChange}) {
    return (
        <ReactAudioPlayer
            src={audioSrc}
            autoPlay
            controls
            onPause={() => handleIsPlaying(false)}
            onPlay={() => handleIsPlaying(true)}
            onSeeked={(e) => handleTimerChange(e)}
        />
    )
}