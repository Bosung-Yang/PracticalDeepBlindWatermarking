import crop
import cutmix
import mixup
import resize
import jpeg
import blur
import gaussian_noise
def main():
    jpeg.main()
    resize.main()
    blur.main()
    crop.crop()
if __name__ == '__main__':
    main()
