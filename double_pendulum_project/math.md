## DOUBLE PENDULUM: Mathematics Work

***
<h3>The following is the calculations file describing the steps in solving the differential equations that represent both pendulums' equations of motion.</h3>

<h7>Let us start with some basic assumptions:
> - Subscript 1 will represent kinematics of motion affecting the first pendulum arm.
> - Subscript 2 will represent kinematics of motion affecting the second pendulum arm.

We start by examining the basic equations of motion in the x- and y-dimension.<br>

Using basic trigonometry, we can induce the basic kinematic equations of position $x_1$, $y_1$, $x_2$, and $y_2$ as functions of the standard pendulum arm angles $\theta_1$ and $\theta_2$. Let $L_1$ and $L_2$ represent the respective lengths of the first and second pendulum arms.</h7>

$$x_1 = L_1~sin~\theta_1$$

$$y_1 = -L_1~cos~\theta_1$$

$$x_2 = x_1~+~L_2~sin~\theta_2$$

$$y_2 = y_1~-~L_2~cos~\theta_2$$

Using basic single-variable calculus, we can calculate the derivatives with respect to time $t$ of each of the position functions to determine the velocity functions in the x- and y-directions for both pendulum arms.

$$v_{x,1} = \tfrac{\partial}{\partial t}(x_1) = L_1~cos~\theta_1$$ (1)

$$v_{y,1} = \tfrac{\partial}{\partial t}(y_1) = \theta_1'~L_1~sin~\theta_1$$ (2)

$$v_{x,2} = \tfrac{\partial}{\partial t}(x_2) = v_{x,1}~+~\theta_2'~L_2~cos~\theta_2$$ (3)

$$v_{y,2} = \tfrac{\partial}{\partial t}(y_2) = v_{y,1}~+~\theta_2'~L_2~sin~\theta_2$$ (4)

One more single-variable derivative of each velocity function with respect to time $t$ will determine the acceleration functions in the x- and y-directions for both pendulum arms.

$$a_{x,1} = \tfrac{\partial^2}{\partial t^2}(x_1) = -\theta_1^{'2}~L_1~sin~\theta_1~+~\theta_1^{''}~L_1~cos~\theta_1$$ (5)

$$a_{y,1} = \tfrac{\partial^2}{\partial t^2}(y_1) = \theta_1^{'2}~L_1~cos~\theta_1~+~\theta_1^{''}~L_1~sin~\theta_1$$ (6)

$$a_{x,2} = \tfrac{\partial^2}{\partial t^2}(x_2) = a_{x,1}~-~\theta_2^{'2}~L_2~sin~\theta_2~+~\theta_2^{''}~L_2~cos~\theta_2$$ (7)

$$a_{y,2} = \tfrac{\partial^2}{\partial t^2}(y_2) = a_{y,1}~+~\theta_2^{'2}~L_2~cos~\theta_2~+~\theta_2^{''}~L_2~sin~\theta_2$$ (8)