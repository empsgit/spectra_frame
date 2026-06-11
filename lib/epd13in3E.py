# /*****************************************************************************
# * | File        :   epd12in48.py
# * | Author      :   Waveshare electrices
# * | Function    :   Hardware underlying interface
# * | Info        :
# *----------------
# * | This version:   V1.0
# * | Date        :   2019-11-01
# * | Info        :   
# ******************************************************************************/
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documnetation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to  whom the Software is
# furished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS OR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
import time
import epdconfig

import PIL
from PIL import Image
import io
import numpy as np

EPD_WIDTH       = 1200
EPD_HEIGHT      = 1600

# 4-bit color codes understood by the panel controller and their RGB values
_PANEL_RGB = np.array([
    [0, 0, 0],        # 0x0 black
    [255, 255, 255],  # 0x1 white
    [255, 255, 0],    # 0x2 yellow
    [255, 0, 0],      # 0x3 red
    [0, 0, 255],      # 0x5 blue
    [0, 255, 0],      # 0x6 green
], dtype=np.int32)
_PANEL_CODES = np.array([0x0, 0x1, 0x2, 0x3, 0x5, 0x6], dtype=np.uint8)

class EPD():
    def __init__(self):
        self.width = EPD_WIDTH
        self.height = EPD_HEIGHT

        self.BLACK  = 0x000000   #   0000  BGR
        self.WHITE  = 0xffffff   #   0001
        self.YELLOW = 0x00ffff   #   0010
        self.RED    = 0x0000ff   #   0011
        self.BLUE   = 0xff0000   #   0101
        self.GREEN  = 0x00ff00   #   0110
        
        self.EPD_CS_M_PIN  = epdconfig.EPD_CS_M_PIN
        self.EPD_CS_S_PIN  = epdconfig.EPD_CS_S_PIN

        self.EPD_DC_PIN  = epdconfig.EPD_DC_PIN
        self.EPD_RST_PIN  = epdconfig.EPD_RST_PIN
        self.EPD_BUSY_PIN  = epdconfig.EPD_BUSY_PIN
        self.EPD_PWR_PIN  = epdconfig.EPD_PWR_PIN


    
    def Reset(self):
        epdconfig.digital_write(self.EPD_RST_PIN, 1) 
        time.sleep(0.03) 
        epdconfig.digital_write(self.EPD_RST_PIN, 0) 
        time.sleep(0.03) 
        epdconfig.digital_write(self.EPD_RST_PIN, 1) 
        time.sleep(0.03) 
        epdconfig.digital_write(self.EPD_RST_PIN, 0) 
        time.sleep(0.03) 
        epdconfig.digital_write(self.EPD_RST_PIN, 1) 
        time.sleep(0.03) 

    def CS_ALL(self, Value):
        epdconfig.digital_write(self.EPD_CS_M_PIN, Value)
        epdconfig.digital_write(self.EPD_CS_S_PIN, Value)

    def SendCommand(self, Command):
        epdconfig.spi_writebyte(Command)

    def SendData(self, Data):
        epdconfig.spi_writebyte(Data)
    
    def SendData2(self, buf, Len):
        epdconfig.spi_writebyte2(buf, Len)

    def ReadBusyH(self, timeout_s=120):
        print("e-Paper busy H")
        start = time.time()
        while(epdconfig.digital_read(self.EPD_BUSY_PIN) == 0):      # 0: busy, 1: idle
            if time.time() - start > timeout_s:
                print("e-Paper busy H TIMEOUT after %ds" % timeout_s)
                raise TimeoutError("EPD busy pin stuck LOW for %ds" % timeout_s)
            epdconfig.delay_ms(5)
        print("e-Paper busy H release")

    def TurnOnDisplay(self):
        print("Write PON")
        self.CS_ALL(0)
        self.SendCommand(0x04)
        self.CS_ALL(1)
        self.ReadBusyH()

        epdconfig.delay_ms(50)

        print("Write DRF")
        self.CS_ALL(0)
        self.SendCommand(0x12)
        self.SendData(0x00)
        self.CS_ALL(1)
        self.ReadBusyH()

        print("Write POF")
        self.CS_ALL(0)
        self.SendCommand(0x02)
        self.SendData(0x00)
        self.CS_ALL(1)
        # Wait for power-off to complete before the caller sends the deep
        # sleep command / cuts SPI; skipping this wedges the controller
        # (frame hangs after Clear)
        self.ReadBusyH()
        print("Display Done!!")

    def Init(self):
        print("EPD init...")
        epdconfig.module_init()
        
        self.Reset()
        # A healthy panel reports idle well under a second after reset;
        # fail fast on a wedged/unpowered panel instead of blocking the
        # display worker for the full 120s refresh timeout
        self.ReadBusyH(timeout_s=20)

        epdconfig.digital_write(self.EPD_CS_M_PIN, 0)
        self.SendCommand(0x74)
        self.SendData(0xC0)
        self.SendData(0x1C)
        self.SendData(0x1C)
        self.SendData(0xCC)
        self.SendData(0xCC)
        self.SendData(0xCC)
        self.SendData(0x15)
        self.SendData(0x15)
        self.SendData(0x55)
        self.CS_ALL(1)

        self.CS_ALL(0)
        self.SendCommand(0xF0)
        self.SendData(0x49)
        self.SendData(0x55)
        self.SendData(0x13)
        self.SendData(0x5D)
        self.SendData(0x05)
        self.SendData(0x10)
        self.CS_ALL(1)

        self.CS_ALL(0)
        self.SendCommand(0x00)
        self.SendData(0xDF)
        self.SendData(0x69)
        self.CS_ALL(1)

        self.CS_ALL(0)
        self.SendCommand(0x50)
        self.SendData(0xF7)
        self.CS_ALL(1)

        self.CS_ALL(0)
        self.SendCommand(0x60)
        self.SendData(0x03)
        self.SendData(0x03)
        self.CS_ALL(1)

        self.CS_ALL(0)
        self.SendCommand(0x86)
        self.SendData(0x10)
        self.CS_ALL(1)

        self.CS_ALL(0)
        self.SendCommand(0xE3)
        self.SendData(0x22)
        self.CS_ALL(1)

        self.CS_ALL(0)
        self.SendCommand(0xE0)
        self.SendData(0x01)
        self.CS_ALL(1)

        self.CS_ALL(0)
        self.SendCommand(0x61)
        self.SendData(0x04)
        self.SendData(0xB0)
        self.SendData(0x03)
        self.SendData(0x20)
        self.CS_ALL(1)

        epdconfig.digital_write(self.EPD_CS_M_PIN, 0)
        self.SendCommand(0x01)
        self.SendData(0x0F)
        self.SendData(0x00)
        self.SendData(0x28)
        self.SendData(0x2C)
        self.SendData(0x28)
        self.SendData(0x38)
        self.CS_ALL(1)

        epdconfig.digital_write(self.EPD_CS_M_PIN, 0)
        self.SendCommand(0xB6)
        self.SendData(0x07)
        self.CS_ALL(1)

        epdconfig.digital_write(self.EPD_CS_M_PIN, 0)
        self.SendCommand(0x06)
        self.SendData(0xE8)
        self.SendData(0x28)
        self.CS_ALL(1)

        epdconfig.digital_write(self.EPD_CS_M_PIN, 0)
        self.SendCommand(0xB7)
        self.SendData(0x01)
        self.CS_ALL(1)

        epdconfig.digital_write(self.EPD_CS_M_PIN, 0)
        self.SendCommand(0x05)
        self.SendData(0xE8)
        self.SendData(0x28)
        self.CS_ALL(1)

        epdconfig.digital_write(self.EPD_CS_M_PIN, 0)
        self.SendCommand(0xB0)
        self.SendData(0x01)
        self.CS_ALL(1)

        epdconfig.digital_write(self.EPD_CS_M_PIN, 0)
        self.SendCommand(0xB1)
        self.SendData(0x02)
        self.CS_ALL(1)
    
    def getbuffer(self, image):
        # Check if we need to rotate the image
        imwidth, imheight = image.size
        if(imwidth == self.width and imheight == self.height):
            image_temp = image
        elif(imwidth == self.height and imheight == self.width):
            image_temp = image.rotate(90, expand=True)
        else:
            raise ValueError("Invalid image dimensions: %d x %d, expected %d x %d" % (imwidth, imheight, self.width, self.height))

        codes = None
        if image_temp.mode == "P":
            # Already palettized by the dithering step: when every palette
            # entry is exactly one of the panel colors, map indices through
            # a LUT instead of re-quantizing all pixels a second time.
            # Palettes with other colors (e.g. the web palette) must NOT be
            # snapped to panel colors without error diffusion - that
            # posterizes the image - so they take the quantize path below.
            pal = image_temp.getpalette()
            pal = np.asarray(pal + [0] * (768 - len(pal)), dtype=np.int32).reshape(256, 3)
            dist = ((pal[:, None, :] - _PANEL_RGB[None, :, :]) ** 2).sum(axis=2)
            if (dist.min(axis=1) == 0).all():
                lut = _PANEL_CODES[dist.argmin(axis=1)]
                codes = lut[np.asarray(image_temp, dtype=np.uint8).ravel()]
        if codes is None:
            # Fallback: convert the source image to the 7 colors, dithering if needed
            pal_image = Image.new("P", (1,1))
            pal_image.putpalette( (0,0,0,  255,255,255,  255,255,0,  255,0,0,  0,0,0,  0,0,255,  0,255,0) + (0,0,0)*249)
            image_7color = image_temp.convert("RGB").quantize(palette=pal_image)
            codes = np.asarray(image_7color, dtype=np.uint8).ravel()

        # PIL does not support 4 bit color, so pack two 4-bit color codes
        # into a single byte to transfer to the panel
        packed = ((codes[0::2] << 4) | (codes[1::2] & 0x0F)).astype(np.uint8)
        return packed.tobytes()

    def buffer_to_image(self, buf):
        """Decode a packed 4-bit display buffer back into a palettized PIL
        image (panel-native portrait orientation). Lets callers preview
        exactly what was sent to the panel."""
        b = np.frombuffer(bytes(buf), dtype=np.uint8)
        codes = np.empty(b.size * 2, dtype=np.uint8)
        codes[0::2] = b >> 4
        codes[1::2] = b & 0x0F
        img = Image.fromarray(codes.reshape((self.height, self.width)), mode='P')
        pal = np.zeros((256, 3), dtype=np.uint8)
        pal[_PANEL_CODES] = _PANEL_RGB
        img.putpalette(pal.ravel().tolist())
        return img
    
    def Clear(self, color=0x11):
        row = bytes([color]) * int(self.width/2)
        epdconfig.digital_write(self.EPD_CS_M_PIN, 0)
        self.SendCommand(0x10)
        for i in range(self.height):
            self.SendData2(row, len(row))
        self.CS_ALL(1)
        epdconfig.digital_write(self.EPD_CS_S_PIN, 0)
        self.SendCommand(0x10)
        for i in range(self.height):
            self.SendData2(row, len(row))
        self.CS_ALL(1)

        self.TurnOnDisplay()

    def display(self, image):
        Width =int(self.width / 4)
        Width1 =int(self.width / 2)

        buf = image if isinstance(image, (bytes, bytearray)) else bytes(image)
        mv = memoryview(buf)

        epdconfig.digital_write(self.EPD_CS_M_PIN, 0)
        self.SendCommand(0x10)
        for i in range(self.height):
            self.SendData2(mv[i * Width1 : i * Width1+Width], Width)
        self.CS_ALL(1)

        epdconfig.digital_write(self.EPD_CS_S_PIN, 0)
        self.SendCommand(0x10)
        for i in range(self.height):
            self.SendData2(mv[i * Width1+Width : i * Width1+Width1], Width)
        self.CS_ALL(1)

        self.TurnOnDisplay()

    def sleep(self):
        self.CS_ALL(0)
        self.SendCommand(0x07)
        self.SendData(0XA5)
        self.CS_ALL(1)

        epdconfig.delay_ms(2000)
        epdconfig.module_exit()
### END OF FILE ###

