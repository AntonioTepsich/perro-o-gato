let tamano = 400;
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let otrocanvas = document.getElementById("otrocanvas");
let ctx = canvas.getContext("2d");
let currentStream = null;
let facingMode = "user";

let modelo = null;

(async() => {
    console.log("Cargando modelo...");
    modelo = await tf.loadLayersModel("model.json");
    console.log("Modelo cargado");
})();

window.onload = function() {
    mostrarCamara();
}

function mostrarCamara() {
    let opciones = {
      audio: false,
      video: {
        width: tamano, height: tamano
      }
    }

    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia(opciones)
          .then(function(stream) {
            currentStream = stream;
            video.srcObject = currentStream;
            procesarCamara();
            predecir();
          })
          .catch(function(err) {
            alert("No se pudo utilizar la camara :(");
            console.log(err);
            alert(err);
          })
    } else {
      alert("No existe la funcion getUserMedia");
    }
}


function procesarCamara() {
    ctx.drawImage(video, 0, 0, tamano, tamano, 0, 0, tamano, tamano);
    setTimeout(procesarCamara, 20);
}


function predecir() {
    if (modelo != null) {
      resample_single(canvas, 90, 90, otrocanvas);

      //Hacer la predicción
      let ctx2 = otrocanvas.getContext("2d");
      let imgData = ctx2.getImageData(0,0, 90, 90);

      let arr = [];
      let arr90 = [];

      for (let p=0; p < imgData.data.length; p+= 4) {
        let rojo = imgData.data[p] / 255;
        let verde = imgData.data[p+1] / 255;
        let azul = imgData.data[p+2] / 255;

        let gris = (rojo+verde+azul)/3;

        arr90.push([gris]);
        if (arr90.length == 90) {
          arr.push(arr90);
          arr90 = [];
        }
      }

      arr = [arr];

      let tensor = tf.tensor4d(arr);
      let resultado = modelo.predict(tensor).dataSync();

      let respuesta;
      if (resultado <= .5) {
        respuesta = "Gato";
      } else {
        respuesta = "Perro";
      }
      document.getElementById("resultado").innerHTML = respuesta;

    }

    setTimeout(predecir, 150);
}

  function resample_single(canvas, width, height, resize_canvas) {
    let width_source = canvas.width;
    let height_source = canvas.height;
    width = Math.round(width);
    height = Math.round(height);

    let ratio_w = width_source / width;
    let ratio_h = height_source / height;
    let ratio_w_half = Math.ceil(ratio_w / 2);
    let ratio_h_half = Math.ceil(ratio_h / 2);

    let ctx = canvas.getContext("2d");
    let ctx2 = resize_canvas.getContext("2d");
    let img = ctx.getImageData(0, 0, width_source, height_source);
    let img2 = ctx2.createImageData(width, height);
    let data = img.data;
    let data2 = img2.data;

    for (let j = 0; j < height; j++) {
        for (let i = 0; i < width; i++) {
            let x2 = (i + j * width) * 4;
            let weight = 0;
            let weights = 0;
            let weights_alpha = 0;
            let gx_r = 0;
            let gx_g = 0;
            let gx_b = 0;
            let gx_a = 0;
            let center_y = (j + 0.5) * ratio_h;
            let yy_start = Math.floor(j * ratio_h);
            let yy_stop = Math.ceil((j + 1) * ratio_h);
            for (let yy = yy_start; yy < yy_stop; yy++) {
                let dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
                let center_x = (i + 0.5) * ratio_w;
                let w0 = dy * dy; //pre-calc part of w
                let xx_start = Math.floor(i * ratio_w);
                let xx_stop = Math.ceil((i + 1) * ratio_w);
                for (let xx = xx_start; xx < xx_stop; xx++) {
                    let dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
                    let w = Math.sqrt(w0 + dx * dx);
                    if (w >= 1) {
                        //pixel too far
                        continue;
                    }
                    //hermite filter
                    weight = 2 * w * w * w - 3 * w * w + 1;
                    let pos_x = 4 * (xx + yy * width_source);
                    //alpha
                    gx_a += weight * data[pos_x + 3];
                    weights_alpha += weight;
                    //colors
                    if (data[pos_x + 3] < 255)
                        weight = weight * data[pos_x + 3] / 250;
                    gx_r += weight * data[pos_x];
                    gx_g += weight * data[pos_x + 1];
                    gx_b += weight * data[pos_x + 2];
                    weights += weight;
                }
            }
            data2[x2] = gx_r / weights;
            data2[x2 + 1] = gx_g / weights;
            data2[x2 + 2] = gx_b / weights;
            data2[x2 + 3] = gx_a / weights_alpha;
        }
    }
    ctx2.putImageData(img2, 0, 0);
}