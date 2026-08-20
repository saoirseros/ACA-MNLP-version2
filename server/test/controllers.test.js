import test from "node:test";
import assert from "node:assert/strict";
import User from "../models/User.js";
import { login } from "../controllers/userController.js";
import { getConversationId } from "../lib/utils.js";

const invokeController = async (controller, body) => {
    let response;
    const res = {
        json(payload) {
            response = payload;
        },
    };

    await controller({ body }, res);
    return response;
};

test("login returns invalid credentials for an unknown email", async () => {
    const originalFindOne = User.findOne;
    User.findOne = async () => null;

    try {
        const response = await invokeController(login, {
            email: "unknown@example.com",
            password: "not-a-real-password",
        });

        assert.deepEqual(response, {
            success: false,
            message: "Invalid credentials",
        });
    } finally {
        User.findOne = originalFindOne;
    }
});

test("conversation ids are independent of participant order", () => {
    assert.equal(
        getConversationId("user-a", "user-b"),
        getConversationId("user-b", "user-a"),
    );
});
