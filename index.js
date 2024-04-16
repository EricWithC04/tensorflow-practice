async function learnLinear () {

    const model = tf.sequential()
    model.add(tf.layers.dense({units: 1, inputShape: [1]}))

    model.compile({
        loss: "meanSquaredError",
        optimizer: "sgd"
    })

    const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1])
    const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1])

    await model.fit(xs, ys, { epochs: 250 })

    document.getElementById("output_field").innerHTML = 
        model.predict(tf.tensor2d([20], [1, 1]))
}

document.getElementById("result").addEventListener("click", learnLinear)