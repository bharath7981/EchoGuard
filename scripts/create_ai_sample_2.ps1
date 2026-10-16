Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 1 # slightly faster speech rate
$synth.SetOutputToWaveFile("$PSScriptRoot\..\data\test_ai_voice_2.wav")
$synth.Speak("Attention. This is an automated artificial intelligence voice simulation. Please verify security authorization.")
$synth.Dispose()
Write-Output "Successfully generated data/test_ai_voice_2.wav"
