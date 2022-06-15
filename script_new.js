Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('https://cdn.jsdelivr.net/gh/HMehta1909/CRMScript/face_recognition_model-weights_manifest.json'),
    faceapi.nets.faceLandmark68Net.loadFromUri('https://cdn.jsdelivr.net/gh/HMehta1909/CRMScript/face_landmark_68_model-weights_manifest.json'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('https://cdn.jsdelivr.net/gh/HMehta1909/CRMScript/ssd_mobilenetv1_model-weights_manifest.json')
]).then(start)

function start() {
    // document.body.append('Models Loaded')
    
    navigator.getUserMedia(
        { video:{} },
        stream => document.getElementById('videoInput').srcObject = stream,
        err => console.error('a'+err)
    )
    
    //video.src = '../videos/speech.mp4'
    console.log('video added')
    recognizeFaces()
}

async function recognizeFaces() {

    const labeledDescriptors = await loadLabeledImages()
    console.log(labeledDescriptors)
    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.7)

    document.getElementById('videoInput').addEventListener('play', async () => {
        console.log('Playing')
        const canvas = faceapi.createCanvasFromMedia(document.getElementById('videoInput'))
        document.body.append(canvas)

        const displaySize = { width: document.getElementById('videoInput').width, height: document.getElementById('videoInput').height }
        faceapi.matchDimensions(canvas, displaySize)

        

        setInterval(async () => {
            const detections = await faceapi.detectAllFaces(document.getElementById('videoInput')).withFaceLandmarks().withFaceDescriptors()

            const resizedDetections = faceapi.resizeResults(detections, displaySize)

            canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)

            const results = resizedDetections.map((d) => {
                return faceMatcher.findBestMatch(d.descriptor)
            })
            results.forEach( (result, i) => {
                if(result["_distance"]<0.5){
                const box = resizedDetections[i].detection.box
                const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
                drawBox.draw(canvas)
                document.getElementById('raisealert').click()
                }
                else{
                    const box = resizedDetections[i].detection.box
                    const drawBox = new faceapi.draw.DrawBox(box, { label: 'Unknown' })
                    drawBox.draw(canvas)    
                }
            })
        }, 100)


        
    })
}


function loadLabeledImages() {
    //const labels = ['Black Widow', 'Captain America', 'Hawkeye' , 'Jim Rhodes', 'Tony Stark', 'Thor', 'Captain Marvel']
    const labels = ['Himanshu', 'Ashutosh Mahajan'] // for WebCam
    return Promise.all(
        labels.map(async (label)=>{
            const descriptions = []
            for(let i=1; i<=2; i++) {
                const img = await faceapi.fetchImage(`../labeled_images/${label}/${i}.jpg`)
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                console.log(label + i + JSON.stringify(detections))
                descriptions.push(detections.descriptor)
            }
            // document.body.append(label+' Faces Loaded | ')
            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}