.. _coordinates:

An Important Note on ``ARTIST`` Coordinates
===========================================

``ARTIST`` uses the east, north, up (ENU) coordinate system in a **four-dimensional** format. To understand these
implications, let us consider two example tensors:

.. code-block:: console

    point_tensor = torch.tensor([e, n, u, 1])
    direction_tensor = torch.tensor([e, n, u, 0])

Both of the above tensors are similar in their first three elements:

- The first element in the above tensors is the **east** coordinate.
- The second element in the above tensors is the **north** coordinate.
- The third element in the above tensors is the **up** coordinate.

However, the fourth element is an extension to a **4D** representation of **3D** coordinates. This enables ``ARTIST``
to perform *rotations* and *translations* within a single *affine transformation matrix*, thus improving efficiency.
With this **4D** representation, it is important to understand:

- The final element in the tensor **is always 1 for points**.
- The final element in the tensor **is always 0 for directions**.
