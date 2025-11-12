//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2025, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  Examples for functional bootstrapping for RLWE ciphertexts using CKKS.
 */

#include "math/hermite.h"
#include "openfhe.h"
#include "schemelet/rlwe-mp.h"

#include <functional>

using namespace lbcrypto;

const BigInteger QBFVINIT(BigInteger(1) << 60);
const BigInteger QBFVINITLARGE(BigInteger(1) << 80);

void IdentityLUT(BigInteger QBFVInit, BigInteger PInput, BigInteger POutput, BigInteger Q, BigInteger Bigq,
                 uint64_t scaleTHI, size_t order, uint32_t numSlots, uint32_t ringDim);

int main() {
    IdentityLUT((BigInteger(1) << (30)), BigInteger(2), (BigInteger(1) << (30)), (BigInteger(1) << (30)),
                (BigInteger(1) << 30), 1, 1, 1024, 4096);
    IdentityLUT((BigInteger(1) << (30)), BigInteger(2), (BigInteger(1) << (20)), (BigInteger(1) << (30)),
                (BigInteger(1) << 30), 1, 1, 1024, 4096);
    IdentityLUT((BigInteger(1) << (30)), BigInteger(2), (BigInteger(1) << (10)), (BigInteger(1) << (30)),
                (BigInteger(1) << 30), 1, 1, 1024, 4096);
    IdentityLUT((BigInteger(1) << (30)), BigInteger(2), (BigInteger(1) << (1)), (BigInteger(1) << (30)),
                (BigInteger(1) << 30), 1, 1, 1024, 4096);
    return 0;
}

void IdentityLUT(BigInteger QBFVInit, BigInteger PInput, BigInteger POutput, BigInteger Q, BigInteger Bigq,
                 uint64_t scaleTHI, size_t order, uint32_t numSlots, uint32_t ringDim) {
    std::function<int64_t(int64_t)> func = [](int64_t x) {
        return x;
    };
    /* 1. Figure out whether sparse packing or full packing should be used.
     * numSlots represents the number of values to be encrypted in BFV.
     * If this number is the same as the ring dimension, then the CKKS slots is half.
     */
    bool flagSP       = (numSlots <= ringDim / 2);  // sparse packing
    auto numSlotsCKKS = flagSP ? numSlots : numSlots / 2;

    /* 2. Input */
    // std::vector<int64_t> x = {1, 0, 0, 1};
    // std::cerr << "First 8 elements of the input (repeated) up to size " << numSlots << ":" << std::endl;
    // std::cerr << x << std::endl;
    // if (x.size() < numSlots)
    //     x = Fill<int64_t>(x, numSlots);

    /* 3. The case of Boolean LUTs using the first order Trigonometric Hermite Interpolation
     * supports an optimized implementation.
     * In particular, it supports real coefficients as opposed to complex coefficients.
     * Therefore, we separate between this case and the general case.
     * There is no need to scale the coefficients in the Boolean case.
     * However, in the general case, it is recommended to scale down the Hermite
     * coefficients in order to bring their magnitude close to one. This scaling
     * is reverted later.
     */
    std::vector<int64_t> coeffint;
    std::vector<std::complex<double>> coeffcomp;
    bool binaryLUT = (PInput.ConvertToInt() == 2) && (order == 1);

    if (binaryLUT) {
        coeffint = {
            func(1),
            func(0) -
                func(1)};  // those are coefficients for [1, cos^2(pi x)], not [1, cos(2pi x)] as in the general case.
    }
    else {
        coeffcomp = GetHermiteTrigCoefficients(func, PInput.ConvertToInt(), order, scaleTHI);  // divided by 2
    }

    /* 4. Set up the cryptoparameters.
     * The scaling factor in CKKS should have the same bit length as the RLWE ciphertext modulus.
     * The number of levels to be reserved before and after the LUT evaluation should be specified.
    * The secret key distribution for CKKS should either be SPARSE_TERNARY or SPARSE_ENCAPSULATED.
    * The SPARSE_TERNARY distribution is for testing purposes as it gives a larger probability of
    * failure but less noise, while the SPARSE_ENCAPSULATED distribution gives a smaller probability
    * of failure at a cost of slightly more noise.
    */
    uint32_t dcrtBits                       = Bigq.GetMSB() - 1;
    uint32_t firstMod                       = Bigq.GetMSB() - 1;
    uint32_t levelsAvailableAfterBootstrap  = 2;
    uint32_t levelsAvailableBeforeBootstrap = 6;
    uint32_t dnum                           = 3;
    SecretKeyDist secretKeyDist             = SPARSE_ENCAPSULATED;
    std::vector<uint32_t> lvlb              = {3, 3};

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetFirstModSize(firstMod);
    parameters.SetNumLargeDigits(dnum);
    parameters.SetBatchSize(numSlotsCKKS);
    parameters.SetRingDim(ringDim);
    uint32_t depth = levelsAvailableAfterBootstrap;

    if (binaryLUT)
        depth += FHECKKSRNS::GetFBTDepth(lvlb, coeffint, PInput, order, secretKeyDist);
    else
        depth += FHECKKSRNS::GetFBTDepth(lvlb, coeffcomp, PInput, order, secretKeyDist);

    parameters.SetMultiplicativeDepth(depth);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    // std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << " and a multiplicative depth of "
    //           << depth << std::endl
    //           << std::endl;

    /* 5. Compute various moduli and scaling sizes, used for scheme conversions.
     * Then generate the setup parameters and necessary keys.
     */
    auto keyPair = cc->KeyGen();

    if (binaryLUT)
        cc->EvalFBTSetup(coeffint, numSlotsCKKS, PInput, POutput, Bigq, keyPair.publicKey, {0, 0}, lvlb,
                         levelsAvailableAfterBootstrap, 0, order);
    else
        cc->EvalFBTSetup(coeffcomp, numSlotsCKKS, PInput, POutput, Bigq, keyPair.publicKey, {0, 0}, lvlb,
                         levelsAvailableAfterBootstrap, 0, order);

    cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlotsCKKS);
    cc->EvalMultKeyGen(keyPair.secretKey);

    /* 6. Perform encryption in the RLWE scheme, using a larger initial ciphertext modulus.
     * Switching the modulus to a smaller ciphertext modulus helps offset the encryption error.
     */
    auto ep = SchemeletRLWEMP::GetElementParams(keyPair.secretKey, depth - (levelsAvailableBeforeBootstrap > 0));

    DCRTPoly::DugType dug;
    DCRTPoly a(dug, ep, Format::EVALUATION);
    a.SetFormat(COEFFICIENT);
    auto ctxtBFV = SchemeletRLWEMP::EncryptCoeffWithZeroB(QBFVInit, a, ep);

    // std::cout << "RLWE ciphertext obtained" << std::endl;
    // std::cout << "Q: " << Q << std::endl;
    // std::cout << "Q/2: " << Q.ConvertToDouble() / 2 << std::endl;
    // std::cout << "POutput: " << POutput << std::endl;
    // std::cout << "bigQPrime: " << bigQPrime << std::endl;

    // std::vector<int64_t> x = {1, 0, 0, 1};
    // if (x.size() < numSlots)
    //     x = Fill<int64_t>(x, numSlots);
    // std::cerr << "First 8 elements of the input (repeated) up to size " << numSlots << ":" << std::endl;
    // std::cerr << x << std::endl;
    // auto ctxtBFV = SchemeletRLWEMP::EncryptCoeff(x, QBFVInit, PInput, keyPair.secretKey, ep);
    // SchemeletRLWEMP::ModSwitch(ctxtBFV, Q, QBFVInit);

    // auto original = SchemeletRLWEMP::DecryptCoeff(ctxtBFV, Q, PInput, keyPair.secretKey, ep, numSlotsCKKS, numSlots);
    // std::cerr << "First 8 elements of the obtained input % PInput: [";
    // std::copy_n(original.begin(), 8, std::ostream_iterator<int64_t>(std::cerr, " "));
    // std::cerr << "]" << std::endl;

    /* 7. Convert from the RLWE ciphertext to a CKKS ciphertext (both use the same secret key).
    */
    auto ctxt = SchemeletRLWEMP::ConvertRLWEToCKKS(*cc, ctxtBFV, keyPair.publicKey, Bigq, numSlotsCKKS,
                                                   depth - (levelsAvailableBeforeBootstrap > 0));

    /* 8. Apply the LUT over the ciphertext.
    */
    Ciphertext<DCRTPoly> ctxtAfterFBT;
    if (binaryLUT)
        ctxtAfterFBT = cc->EvalFBT(ctxt, coeffint, PInput.GetMSB() - 1, ep->GetModulus(), scaleTHI, 0, order);
    else
        ctxtAfterFBT = cc->EvalFBT(ctxt, coeffcomp, PInput.GetMSB() - 1, ep->GetModulus(), scaleTHI, 0, order);

    /* 9. Convert the result back to RLWE.
    */
    auto polys = SchemeletRLWEMP::ConvertCKKSToRLWE(ctxtAfterFBT, Q);

    auto computed =
        SchemeletRLWEMP::DecryptCoeffWithoutRound(polys, Q, keyPair.secretKey, ep, numSlotsCKKS, numSlots, false);

    auto delta = Q / POutput;
    auto half  = (Q >> 1);
    std::cerr << "Q/POutput (0 centered): " << ((delta > half) ? -(Q - delta) : delta) << std::endl;
    std::cerr << "First 8 elements of the obtained output: [";
    std::copy_n(computed.begin(), 8, std::ostream_iterator<int64_t>(std::cerr, " "));
    std::cerr << "]" << std::endl;
}
