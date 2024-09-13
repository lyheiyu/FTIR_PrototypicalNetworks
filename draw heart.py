# Import turtle package
import turtle
import colorsys
# Creating a turtle object(pen)
pen = turtle.Turtle()
t = turtle.Turtle()
s = turtle.Screen()
# Defining a method to draw curve

s.bgcolor('red')
t.speed(200)
def curve():
	for i in range(10):

		# Defining step by step curve motion
		pen.right(10)
		pen.forward(10)

# Defining method to draw a full heart
def star():
	# import for turtle

	pen.fillcolor('yellow')
	pen.begin_fill()
	t.color('red')
	# executing loop 5 times for a star
	for i in range(5):
		# moving turtle 100 units forward
		pen.forward(100)
		# rotating turtle 144 degree right
		pen.right(144)
	pen.end_fill()
	pen.fillcolor(0,0,0)
	pen.begin_fill()
	pen.left(30)
	pen.forward(100)
	pen.end_fill()
	pen.fillcolor('yellow')
	pen.begin_fill()
	for i in range(5):
		# moving turtle 100 units forward
		pen.forward(50)

		# rotating turtle 144 degree right
		pen.right(144)
	pen.end_fill()
	for i in range(3):
		pen.fillcolor('red')
		pen.begin_fill()
		pen.right(50)
		pen.forward(50)
		pen.end_fill()
		pen.fillcolor('yellow')
		pen.begin_fill()
		for i in range(5):
			# moving turtle 100 units forward
			pen.forward(50)

			# rotating turtle 144 degree right
			pen.right(144)
		pen.end_fill()

def heart():
	n = 30
	h = 0
	# Set the fill color to red
	for i in range(30):
		c = colorsys.hsv_to_rgb(h, 1, 0.8)
		h += 1/ n
		# t.color(c)
		pen.fillcolor(c)

	# Start filling the color
		pen.begin_fill()

	# Draw the left line
	# 	pen.left(140)
	# 	pen.forward(113)

	# Draw the left curve
		curve()
		# pen.left(120)
		pen.forward(-20)
	# Draw the right curve
		curve()
		pen.forward(20)
	# Draw the right line
	# 	pen.forward(112)

	# Ending the filling of the color
		pen.end_fill()

# Defining method to write text


def txt():

	# Move turtle to air
	pen.up()

	# Move turtle to a given position
	pen.setpos(-68, 95)

	# Move the turtle to the ground
	pen.down()

	# Set the text color to lightgreen
	pen.color('lightgreen')

	# Write the specified text in
	# specified font style and size
	pen.write("LOVE YOU", font=(
	"Verdana", 12, "bold"))


# Draw a heart

star()
# Write text
txt()

# To hide turtle
# pen.ht()
