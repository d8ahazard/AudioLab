import WaveSurfer from '/gradio_api/file=scripts/wavesurfer.esm.js'

class WaveSurferPlayer {
  constructor(player_container) {
    this.player_container = player_container

    this.playButton = player_container.querySelector('#play_button')
    this.playContainer = this.playButton.querySelector('#play')
    this.pauseContainer = this.playButton.querySelector('#pause')

    this.wavesurfer = this.createPlayer(player_container)
  }

  loadAudioEventHandler(e) {
    this.wavesurfer.load(e.detail.url)
    this.audio_file_url = e.detail.url
    this.wavesurfer.setTime(0)
  }

  wavesurferPlayEventHandler() {
    this.enablePlayButton(false)

    const event = new CustomEvent("pauseallplayers", {
      detail : {
        exclude_player: this.player_container
      },
    })

    this.dispatchToPlayers(event)
  }

  pauseAllPlayers(e) {
    if (e.detail.exclude_player != this.player_container) {
      this.wavesurfer.pause()
    }    
  }

  dowloadAudioEventHandler() {
    const tmpElem = document.createElement("a")
    const url = this.audio_file_url
    tmpElem.href=url
    tmpElem.download=url.split('/').pop()
    document.body.appendChild(tmpElem)
    tmpElem.click()
    tmpElem.remove()
  }

  enablePlayButton(enabled) {
    if (enabled) {
      this.playContainer.style.visibility = "visible";
      this.pauseContainer.style.visibility = "hidden";
    }
    else{
      this.playContainer.style.visibility = "hidden";
      this.pauseContainer.style.visibility = "visible";
    }
  }

  dispatchToPlayers(event) {
    const elements = document.querySelectorAll("div[id^='wavesurfer_player']")
    elements.forEach((e)=> {
      e.dispatchEvent(event)
    })    
  }

  createPlayer(player_container) {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    
    // Define the waveform gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height * 1.35)
    gradient.addColorStop(0, '#656666') // Top color
    gradient.addColorStop((canvas.height * 0.7) / canvas.height, '#656666') // Top color
    gradient.addColorStop((canvas.height * 0.7 + 1) / canvas.height, '#ffffff') // White line
    gradient.addColorStop((canvas.height * 0.7 + 2) / canvas.height, '#ffffff') // White line
    gradient.addColorStop((canvas.height * 0.7 + 3) / canvas.height, '#B1B1B1') // Bottom color
    gradient.addColorStop(1, '#B1B1B1') // Bottom color
    
    // Define the progress gradient
    const progressGradient = ctx.createLinearGradient(0, 0, 0, canvas.height * 1.35)
    progressGradient.addColorStop(0, '#EE772F') // Top color
    progressGradient.addColorStop((canvas.height * 0.7) / canvas.height, '#EB4926') // Top color
    progressGradient.addColorStop((canvas.height * 0.7 + 1) / canvas.height, '#ffffff') // White line
    progressGradient.addColorStop((canvas.height * 0.7 + 2) / canvas.height, '#ffffff') // White line
    progressGradient.addColorStop((canvas.height * 0.7 + 3) / canvas.height, '#F6B094') // Bottom color
    progressGradient.addColorStop(1, '#F6B094') // Bottom color
       
    const wavesurfer = WaveSurfer.create({
      container: player_container,
      waveColor: gradient,
      progressColor: progressGradient,
      barWidth: 2,
      mediaControls: false,
      dragToSeek: true,
      hideScrollbar: true,
      height: 130,
    })    

    {
      const downloadButton = player_container.querySelector('#download_button')
      downloadButton.addEventListener("click", ()=> this.dowloadAudioEventHandler())

      player_container.addEventListener("load", (e)=>this.loadAudioEventHandler(e))
      player_container.addEventListener("pauseallplayers", (e)=>this.pauseAllPlayers(e))
      this.playButton.addEventListener("click", ()=> wavesurfer.playPause())
      wavesurfer.on('interaction', () => wavesurfer.play())

      wavesurfer.on('play', () => this.wavesurferPlayEventHandler())
      wavesurfer.on('pause', () => this.enablePlayButton(true))
    }

    // Time & duration
    {      
      const timeEl = player_container.querySelector('#time')
      const durationEl = player_container.querySelector('#duration')
      wavesurfer.on('decode', (duration) => (durationEl.textContent = this.formatTime(duration)))
      wavesurfer.on('timeupdate', (currentTime) => (timeEl.textContent = this.formatTime(currentTime)))
    }
    return wavesurfer
  }

  formatTime(seconds) {
    const minutes = Math.floor(seconds / 60)
    const secondsRemainder = Math.round(seconds) % 60
    const paddedSeconds = `0${secondsRemainder}`.slice(-2)
    return `${minutes}:${paddedSeconds}`
  }
}

class AudioPlayerContainer {
  constructor() {
    this.players = new Map()
    this.waitForElements("div[id^='wavesurfer_player']").then((elements) =>{
      elements.forEach((elem) => {
        this.players[elem.id] = new WaveSurferPlayer(elem)
      })
    })
  }

  waitForElements(selector) {
    return new Promise(resolve => {
      const elements = document.querySelectorAll(selector)
      if (elements.length != 0) {
        return resolve(elements)
      }
      const observer = new MutationObserver(mutations => {
        const elements = document.querySelectorAll(selector)
        if (elements.length != 0) {
          observer.disconnect()
          return resolve(elements)
        }  
      })
      observer.observe(document.body, {
        childList: true,
        subtree: true
      })
    })
  }
}

const players = new AudioPlayerContainer()
