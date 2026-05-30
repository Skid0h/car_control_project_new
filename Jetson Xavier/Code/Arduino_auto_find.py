import serial.tools.list_ports

def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if ('Arduino' in port.description or 'CH340' in port.description or 'USB Serial' in port.description):
            return port.device
       
    for port in ports:
        if port.vid and port.pid:
            if (port.vid == 0x2341) or (port.vid == 0x1A86):
                print (f"Ардуино найдена на порту: {port.device}")
                return port.device
    print('Arduino not found')
    return None