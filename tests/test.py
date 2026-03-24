"""
step1 - compute quantiles over the selected pages -> bounds
step2 - check corpus against empirical quantiles -> loop over the pages in the corpus and check if there's any image that meets the template standards


NOTE:
now we consider baricentes of the image and sizes but vertices could be used as well (verteces add information about the shape of the image, namely, if it is rectangular or squared)
"""
# vc =

# x_upper_left
# y_upper_left
# size_upper_left

# x_centered
# y_centered
# size_centered

# x_bottom_right
# y_bottom_right
# size_bottom_right

# ~30 examples -> q_alpha

# bounds_x_upper_left = (x_upper_left_lower_bound, x_upper_left_upper_bound)
bounds_x_upper_left = (0, 10)


# y_upper_left lower_bound
# size_upper_left lower_bound


# y_upper_left upper_bound
# size_upper_left upper_bound


# x_centered lower_bound
# y_centered lower_bound
# size_centered lower_bound

# x_centered upper_bound
# y_centered upper_bound
# size_centered upper_bound


# x_bottom_right lower_bound
# y_bottom_right lower_bound
# size_bottom_right lower_bound

# x_bottom_right upper_bound
# y_bottom_right upper_bound
# size_bottom_right upper_bound

corpus = {}

image_upper_left_check = False
image_lower_right_check = False
image_centered_check = False

for page in corpus:

    if len(page.images) == 3:

        for image in page.images:

            if (
                image.x in bounds_x_upper_left
                and image.y in (0, 20)
                and image.size in (0, 10)
            ):
                image_upper_left_check = True
            elif (
                image.x in bounds_x_upper_left
                and image.y in (0, 20)
                and image.size in (0, 10)
            ):
                image_lower_right_check = True
            elif (
                image.x in bounds_x_upper_left
                and image.y in (0, 20)
                and image.size in (0, 10)
            ):
                image_centered_check = True
            else:
                continue

    else:
        continue

    if (
        image_upper_left_check
        and image_lower_right_check
        and image_centered_check
    ):
        print(f"the page meets standard for the template {page.id}")
    else:
        print("try another one")
