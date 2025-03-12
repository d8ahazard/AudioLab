globalThis.autoDownloadData = (src) => {
    const dst=src.split('/').pop()
    fetch("gradio_api/file="+src)
    .then((res)=> {
        if (!res.ok) {
            throw new Error("Can't download file!")
        }
        return res.blob()
    })
    .then((file)=> {
        let tmpUrl = URL.createObjectURL(file)
        const tmpElem = document.createElement("a")
        tmpElem.href=tmpUrl
        tmpElem.download=dst
        document.body.appendChild(tmpElem)
        tmpElem.click()
        URL.revokeObjectURL(tmpUrl)
        tmpElem.remove()
    })
}
