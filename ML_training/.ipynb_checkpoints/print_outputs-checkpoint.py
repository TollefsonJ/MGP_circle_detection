# Print resized versions of false positives and false negatives


for i in range(0,1336):
    if y_test[i] == 0 and y_pred[i] == 1:
        img = X_test[i]
        img2 = np.reshape(img, (64,64))
        path = "training_images/" + set + "/output/resized/falsepos/"+file[i]
        cv2.imwrite(path, img2)
    else:
        if y_test[i] == 1 and y_pred[i] == 0:
            img = X_test[i]
            img2 = np.reshape(img, (64,64))
            path = "training_images/" + set + "/output/resized/falseneg/"+file[i]
            cv2.imwrite(path, img2)
        else:
            continue