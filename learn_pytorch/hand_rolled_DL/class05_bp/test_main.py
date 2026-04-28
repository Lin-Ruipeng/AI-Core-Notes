import main


def test_mullayer():
    apple = 100
    apple_num = 2
    tax = 1.1

    # layer
    mul_apple_layer = main.MulLayer()
    mul_tax_layer = main.MulLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    # price(price)
    assert abs(price - 220) < 1e-3

    # backward
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    assert abs(dapple - 2.2) < 1e-3
    assert abs(dapple_num - 110) < 1e-3
    assert abs(dtax - 200) < 1e-3


def test_add():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    # layer
    mul_apple_layer = main.MulLayer()
    mul_orange_layer = main.MulLayer()
    add_apple_orange_layer = main.AddLayer()
    mul_tax_layer = main.MulLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
    orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
    price = mul_tax_layer.forward(all_price, tax)  # (4)

    # backward
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)
    assert abs(dtax - 650) < 1e-4
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)
    assert abs(dorange_num - 165) < 1e-4
    assert abs(dorange - 3.3) < 1e-4
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)
    assert abs(dapple - 2.2) < 1e-4
    assert abs(dapple_num - 110) < 1e-4
