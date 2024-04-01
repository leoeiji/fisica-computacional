function Sketch(p) {
  p.setup = function () {
    p.createCanvas(400, 400);
  };

  p.draw = function () {
    p.background(255);
    p.ellipse(p.width / 2, p.height / 2, 50, 50);
  };
}

export default Sketch;
