import org.apache.spark.ml.linalg.{Vector, Vectors}

object TransformUtils {

  case class TrainRecord(label: Double, features: Vector)

  val (xx, yy, zz) = (21, 21, 7)

  // these functions are used for calculate the coordinates after rotation
  def rotateMapping = Map[Int, (Int, Int) => Int](90 -> ((px, py) => (yy - py - 1) * xx + px),
    180 -> ((px, py) => (xx - px - 1) * yy + (yy - py - 1)),
    270 -> ((px, py) => py * xx + (xx - px - 1)))

  // mirror the image and create new neighbors array based on given neighbors arary
  // degree means how many degrees the original image has rotated
  def mirror(neighbors: Vector, degree: Int): Vector = {
    if (degree != 0 && degree != 90 && degree != 180 && degree != 270)
      throw new IllegalArgumentException("degree should be 0, 90, 180 or 270")
    val res = Array.ofDim[Double](neighbors.size)
    for (px <- 0 until xx) {
      for (py <- 0 until yy) {
        val srcStart = if (degree % 180 == 0) (px * yy + py) * zz else (py * xx + px) * zz
        val dstStart = if (degree % 180 == 0) ((xx - px - 1) * yy + py) * zz else ((yy - py - 1) * xx + px) * zz
        for (i <- 0 until zz) res(i + dstStart) = neighbors(i + srcStart)
      }
    }
    Vectors.dense(res)
  }

  // rotate the image by degree, and create new neighbors array
  def rotate(neighbors: Vector, degree: Int): Vector = {
    if (degree != 90 && degree != 180 && degree != 270)
      throw new IllegalArgumentException("degree should be 90, 180 or 270")
    val res = Array.ofDim[Double](neighbors.size)
    for (px <- 0 until xx) {
      for (py <- 0 until yy) {
        val srcStart = (px * yy + py) * zz
        val dstStart = rotateMapping(degree)(px, py) * zz
        for (i <- 0 until zz) res(i + dstStart) = neighbors(i + srcStart)
      }
    }
    Vectors.dense(res)
  }
}