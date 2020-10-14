# MGP_circle_detection

This is a project at Brown University to identify coal gas infrastructure in historic fire insurance maps, taking advantage of the unique circular structure associated with historic gasometers and gas holders.

We use the Hough.Circles package to identify circular features in Sanborn maps; circles are passed through an MLP classifier to predict whether they represent coal gas infrastructure.
