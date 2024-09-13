num_new_classes=10
classes=[1,2,3,4,5,6,7,8,9]

classes += [i + len(classes) for i in range(num_new_classes)]

print(classes)
x=[i + len(classes) for i in range(num_new_classes)]
print(x)