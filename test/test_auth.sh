#!/bin/bash

# Configuration
URL="http://localhost:3000"
COOKIE_JAR="cookies.txt"
USERNAME="user_test_$(date +%s)" 
PASSWORD="password12345678"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' 

echo "-----------------------------------------------------"
echo "FULL FLOW TEST: $USERNAME"
echo "-----------------------------------------------------"

rm -f $COOKIE_JAR

# 1. REGISTER
echo -n "[1] Registering... "
curl -s -X POST "$URL/register" \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}" > /dev/null
echo -e "${GREEN}OK${NC}"

# 2. LOGIN (With Remember Me)
echo -n "[2] Login (Remember Me)... "
curl -s -c $COOKIE_JAR -X POST "$URL/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\", \"remember_me\": true}" > /dev/null

if grep -q "remember_me" $COOKIE_JAR; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC} (Cookie missing)"
    exit 1
fi

# 3. VERIFY WE ARE LOGGED IN
echo -n "[3] Check /me (Initial)... "
RESP=$(curl -s -b $COOKIE_JAR "$URL/me")
if echo "$RESP" | grep -q "$USERNAME"; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

# 4. SIMULATE SESSION DEATH
# We delete the line containing 'id' (the session cookie) from the jar.
# We keep 'remember_me'.
echo -n "[4] Killing Session Cookie... "
sed -i.bak '/id/d' $COOKIE_JAR 2>/dev/null || sed -i '' '/id/d' $COOKIE_JAR # sed compatibility
echo -e "${GREEN}Done${NC}"

# Verify we really killed it (grep should find nothing)
if grep -q "id" $COOKIE_JAR; then
    echo -e "${RED}ERROR: Session cookie still exists!${NC}"
    exit 1
fi

# 5. RESTORE SESSION (The Magic Moment)
# We request /me. The server sees no session, checks remember_me, matches DB,
# creates a NEW session, and rotates the remember token.
echo -n "[5] Attempting Auto-Login via Cookie... "
curl -s -b $COOKIE_JAR -c $COOKIE_JAR "$URL/me" > /dev/null

# We check if the server gave us a NEW session cookie (id)
if grep -q "id" $COOKIE_JAR; then
    echo -e "${GREEN}SUCCESS!${NC}"
    echo "    (Server accepted remember_me and issued new session)"
else
    echo -e "${RED}FAILED${NC}"
    echo "    (Server did not issue a new session. Check server logs.)"
    exit 1
fi

# 6. FINAL CHECK
echo -n "[6] Final Auth Check... "
RESP=$(curl -s -b $COOKIE_JAR "$URL/me")
if echo "$RESP" | grep -q "$USERNAME"; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo "-----------------------------------------------------"
echo "ALL TESTS PASSED"
rm -f $COOKIE_JAR $COOKIE_JAR.bak
