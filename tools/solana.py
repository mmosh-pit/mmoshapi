import re
import os
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey as PublicKey
from solders.transaction import Transaction
from solders.system_program import transfer, TransferParams
from spl.token.constants import TOKEN_PROGRAM_ID
from langchain.tools import tool
from solders.system_program import transfer, TransferParams
import base58
from solders.instruction import Instruction, AccountMeta
from solders.hash import Hash
from solders.rpc.responses import GetLatestBlockhashResp
from solders.message import MessageV0
from solders.system_program import ID as SYSTEM_PROGRAM_ID
from solders.transaction import VersionedTransaction
from spl.token.instructions import transfer_checked, get_associated_token_address, TransferCheckedParams
from pydantic import BaseModel, Field

class SolanaTransferInput(BaseModel):
    amount: float = Field(..., description="Amount to send")
    destination: str = Field(..., description="Destination wallet address")
    token: str = Field(..., description="Token to send, e.g., SOL or MMOSH")

LAMPORTS_PER_SOL = 10**9

TOKEN_MINTS = {
    "MMOSH": "6vgT7gxtF8Jdu7foPDZzdHxkwYFX9Y1jvgpxP8vH2Apw",
}

base58_key = os.getenv("SENDER_PRIVATE_KEY_HEX")

@tool("solana_transfer", args_schema=SolanaTransferInput)
async def solana_transfer(*, amount: float, destination: str, token: str) -> str:
    """Transfer SOL or MMOSH to a destination address on Solana."""
    client = AsyncClient("https://api.devnet.solana.com")

    key_bytes = base58.b58decode(base58_key)
    if len(key_bytes) != 64:
        raise ValueError("SENDER_PRIVATE_KEY_HEX must be a 64-byte hex string")

    # Create keypair
    sender = Keypair.from_bytes(key_bytes)
    sender_pubkey = sender.pubkey()
    to_pubkey = PublicKey.from_string(destination) 

    if token.upper() == "SOL":
        ix = transfer(
            TransferParams(
                from_pubkey=sender_pubkey, to_pubkey=to_pubkey, lamports=int(amount * 10**9)
            )
        )
        blockhash_resp = await client.get_latest_blockhash()
        recent_blockhash = blockhash_resp.value.blockhash
        msg = MessageV0.try_compile(
            payer=sender_pubkey,
            instructions=[ix],
            address_lookup_table_accounts=[],
            recent_blockhash=recent_blockhash,
        )
        tx = VersionedTransaction(msg, [sender])
        response = await client.send_transaction(tx)
    else:

        mint_address = TOKEN_MINTS.get(token.upper())
        if not mint_address:
            await client.close()
            return f"❌ Unsupported token {token}."

        mint = PublicKey.from_string(mint_address)
        sender_token_account = get_associated_token_address(sender_pubkey, mint)
        recipient_token_account = get_associated_token_address(to_pubkey, mint)

        decimals = 9

        # Create transfer instruction
        ix = transfer_checked(
            TransferCheckedParams(
                source=sender_token_account,
                mint=mint,
                dest=recipient_token_account,
                owner=sender_pubkey,
                amount=int(amount * 10**decimals),
                decimals=decimals,
                program_id=TOKEN_PROGRAM_ID,
            )
        )
                
        blockhash_resp = await client.get_latest_blockhash()
        recent_blockhash = blockhash_resp.value.blockhash
        msg = MessageV0.try_compile(
            payer=sender_pubkey,
            instructions=[ix],
            address_lookup_table_accounts=[],
            recent_blockhash=recent_blockhash,
        )
        tx = VersionedTransaction(msg, [sender])
        response = await client.send_transaction(tx)

    await client.close()
    return f"✅ Transfer successful. Transaction signature: {str(response.value)}"