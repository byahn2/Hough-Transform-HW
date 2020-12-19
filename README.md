# Hough-Transform-HW
Implement a Python routine for boundary shapes and apply it to image of shoe print.

The Hough transform was discussed in class as a way to detect
lines in images by voting in a transformed parameter space. For the case of lines, this
is a 2D parameter space. Now, consider different possible parameterizations of
boundaries of interest. In particular, consider a circle or a quadratic curve. These
examples may be used in various applications. In this problem, we will use shoe-print
images as examples.

1. Select either the circle or quadratic curve and derive the Hough transform for that
case.

  I selected the circle for which the parameter space is a,b,r where a represents
  the x position or row of the center of the circle, b represents the y position or
  column of the center of the circle, and r represents the radius of the circle.

2. Implement your own Python routine for the boundary shape you chose and apply
it to the appropriate image in shoeprint/. You may need to think behind simple
Canny edge detection to provide sufficient input to the Hough transform.
  See code and images.
  
3. Are you able to get it to work? Discuss the benefits and drawbacks of the Hough
transform in the context of the boundary shape you chose and the image domain
you worked with. Supplement the discussion with other images if needed. Max
1 page.

  The shoe image is result of drawing all of the circles from the parameter bins which
  received 85% or more of the maximum number of votes. This does a decent job of
  identifying the main circles of interest. However, the concentric circles are somewhat
  difficult, because it seems to match the inner edge of a circular band with its outer edge
  on the other side, producing a circle that is approximately the correct size, but not
  centered properly. This also happened in a less noisy image without concentric circles
  but with a lower threshold. Another issue was the noise in the image. The image is very
  noisy and has some circular points of noise. In order to identify the important circles, I
  had to adjust the canny edge detector to make it subdue most of the information in the
  image.
  
  Using the Hough transform was good because it generates a perfect circle of the radius
  and center determined, so as long as enough of the circle’s perimeter is there to vote for
  the correct center, it can identify the circle even if the circle boundary itself is broken or
  noisy. This is particularly useful for noise images or circles that are partially occluded.
  However, it also means that if the circle is actually an ellipse, it will get a lot of votes for
  the center of the ellipse, but will likely draw circles with different radii. This is
  demonstrated in the circle images.
  
  The final point I would like to talk about is that I had to choose my parameters according
  to the result I wanted. I needed to treat noisy images differently. I also had to use
  different thresholds for images with different numbers of circles. In the first image, there
  were only 3 circles, so I took the top 15% of detected circles, whereas in the circle image
  above and to the left, there are many circles, so I took the top 30% of detected circles.
  As a separate note, I really enjoyed this project. I implemented my own canny edge
  detector and learned a lot!
